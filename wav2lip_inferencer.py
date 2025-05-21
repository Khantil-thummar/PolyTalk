import numpy as np
import scipy, cv2, os, sys, audio
import subprocess
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip
import platform

class Wav2LipInferencer:
    def __init__(self, checkpoint_path,
                 img_size=96,
                 face_det_batch_size=16,
                 wav2lip_batch_size=128,
                 pads=[0, 10, 0, 0],
                 resize_factor=1,
                 crop=[0, -1, 0, -1],
                 box=[-1, -1, -1, -1],
                 rotate=False,
                 nosmooth=False,
                 device=None):
        self.checkpoint_path = checkpoint_path
        self.img_size = img_size
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.pads = pads
        self.resize_factor = resize_factor
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference.')
        self.model = self.load_model(self.checkpoint_path)
        self.mel_step_size = 16
        # Reuse face detector instance
        self.face_detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, flip_input=False, device=self.device)

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        print(f"Load checkpoint from: {path}")
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        model = model.to(self.device)
        return model.eval()

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        batch_size = self.face_det_batch_size
        detector = self.face_detector
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])
        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        return results

    def datagen(self, frames, mels, static, box):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        if box[0] == -1:
            if not static:
                face_det_results = self.face_detect(frames)
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        for i, m in enumerate(mels):
            idx = 0 if static else i%len(frames)
            frame_to_save = frames[idx]  # .copy() not needed
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (self.img_size, self.img_size))
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch

    def infer(self, face_path, audio_path, outfile='results/result_voice.mp4', fps=25., static=None):
        # Determine if static
        if static is None:
            static = os.path.isfile(face_path) and face_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']
        if static:
            full_frames = [cv2.imread(face_path)]
        else:
            video_stream = cv2.VideoCapture(face_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS) or fps
            print('Reading video frames...')
            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//self.resize_factor, frame.shape[0]//self.resize_factor))
                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
        print(f"Number of frames available for inference: {len(full_frames)}")
        # Ensure audio is wav
        temp_wav = None
        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            os.makedirs('temp', exist_ok=True)
            temp_wav = 'temp/temp.wav'
            command = f'ffmpeg -y -i {audio_path} -strict -2 {temp_wav}'
            subprocess.call(command, shell=True)
            audio_path = temp_wav
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        print(f"Length of mel chunks: {len(mel_chunks)}")
        # Adjust frames to mel chunks
        full_frames = full_frames[:len(mel_chunks)]
        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames, mel_chunks, static, self.box)
        out = None
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                os.makedirs('temp', exist_ok=True)
                out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            # Use inference_mode if available
            if hasattr(torch, 'inference_mode'):
                context = torch.inference_mode()
            else:
                context = torch.no_grad()
            with context:
                pred = self.model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
        if out is not None:
            out.release()
        command = f'ffmpeg -y -i {audio_path} -i temp/result.avi -strict -2 -q:v 1 {outfile}'
        subprocess.call(command, shell=platform.system() != 'Windows')
        # Release GPU memory
        del img_batch, mel_batch, pred
        torch.cuda.empty_cache()
        # Clean up temp wav if created
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        print(f"Saved output to {outfile}") 