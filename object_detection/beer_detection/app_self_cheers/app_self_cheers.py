import time
import cv2
from PIL import Image
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from collections import namedtuple
import pygame.mixer

MODEL_FILENAME = "beer_mobilenet_v2_quantized_300x300_edgetpu.tflite"
LABEL_FILENAME = "beer_label.txt"

def cv2pil(image_cv):
	image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
	image_pil = Image.fromarray(image_cv)
	image_pil = image_pil.convert('RGB')
	return image_pil

Rectangle = namedtuple('Rectangle', 'x0 y0 x1 y1 center_x center_y speed_x speed_y')
class RoughTracker():
	def __init__(self):
		self.MAX_FRAME = 15
		self.MAX__NO_DETFRAME = 10
		self.frameIndex = 0
		self.track = []
		self.num_no_detection = 0
	
	def clear(self):
		self.track.clear()

	def update(self, x0 = None, y0 = None, x1 = None, y1 = None):
		if x0 is None:
			self.num_no_detection = self.num_no_detection + 1
			if self.num_no_detection > self.MAX__NO_DETFRAME or len(self.track) == 0:
				self.num_no_detection = 0
				self.track.clear()
				return 0, 0
			# use the previous result for no detection frame
			x0 = self.track[-1].x0
			y0 = self.track[-1].y0
			x1 = self.track[-1].x1
			y1 = self.track[-1].y1

		center_x = (x1 + x0) / 2
		center_y = (y1 + y0) / 2
		speed_x = 0
		speed_y = 0
		if len(self.track) > self.MAX_FRAME:
			# remove the oldest one
			rect_past = self.track.pop(0)
			# calculate speed
			speed_x = center_x - rect_past.center_x
			speed_y = center_y - rect_past.center_y
		rect = Rectangle(x0, y0, x1, y1, center_x, center_y, speed_x, speed_y)
		self.track.append(rect)
		return speed_x, speed_y

if __name__ == '__main__':
	# Load model and prepare TPU engine
	engine = DetectionEngine(MODEL_FILENAME)
	labels = dataset_utils.read_label_file(LABEL_FILENAME)

	# Initialize camera input
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	# Create tracker
	tracker = RoughTracker()

	# Initialize mixer
	pygame.mixer.init()
	pygame.mixer.music.load("yukumo_0001.mp3")
	
	while True:
		start = time.time()
		# capture image
		ret, img_org = cap.read()
		pil_img = cv2pil(img_org)

		# Run inference
		ans = engine.detect_with_image(pil_img, threshold=0.2, keep_aspect_ratio=False, relative_coord=True, top_k=5)

		# Retrieve results
		# print ('-----------------------------------------')
		if ans:
			for obj in ans:
				box = obj.bounding_box.flatten().tolist()
				# print(labels[obj.label_id] + "({0:.3f}) ".format(obj.score)
				#  + "({0:.3f}, ".format(box[0]) + "{0:.3f}, ".format(box[1]) + "{0:.3f}, ".format(box[2]) + "{0:.3f})".format(box[3]))
				x0 = int(box[0] * img_org.shape[1])
				y0 = int(box[1] * img_org.shape[0])
				x1 = int(box[2] * img_org.shape[1])
				y1 = int(box[3] * img_org.shape[0])
				cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
				cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
				cv2.putText(img_org,
						str(labels[obj.label_id]),
						(x0, y0),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,
						(255, 255, 255),
						2)
				
				# Check cheers
				# assume there is only one object
				speed_x, speed_y = tracker.update(x0, y0, x1, y1)
				if speed_y < -80:
					print("cheers")
					pygame.mixer.music.play(1)
					tracker.clear()

		else:
			_, _ = tracker.update()
		# Draw the result
		cv2.imshow('image', img_org)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		elapsed_time = time.time() - start
		# print('inference time = ', "{0:.2f}".format(engine.get_inference_time()) , '[msec]')
		# print('total time = ', "{0:.2f}".format(elapsed_time * 1000), '[msec] (', "{0:.2f}".format(1 / elapsed_time), ' fps)')

	cv2.destroyAllWindows()
	pygame.mixer.music.stop()
