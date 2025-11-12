import numpy as np
import ffmpeg

def save_video(file_name:str, frames:list, framerate:float=30.0,frametime:np.ndarray | float=5,vcodec='libx264'):
    if isinstance(frametime,float):
        frametime = frametime*np.ones(len(frames))
    height,width,channels = frames[0].shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height), r=framerate)
            .output(file_name, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True, overwrite_output=True, pipe_stderr=True)
    )
    for i, frame in enumerate(frames):
        n_frames = frametime[i]*framerate

        while n_frames >0:
            try:
                process.stdin.write(
                    frame.astype(np.uint8).tobytes()
                )
                n_frames -= 1
            except Exception as e: # should probably be an exception related to process.stdin.write
                for line in io.TextIOWrapper(process.stderr, encoding="utf-8"): # I didn't know how to get the stderr from the process, but this worked for me
                    print(line) # <-- print all the lines in the processes stderr after it has errored
                process.stdin.close()
                process.wait()
    return

