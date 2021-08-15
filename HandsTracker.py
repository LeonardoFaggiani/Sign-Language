import cv2
import mediapipe as mp


class handDetector():
  def __init__(self, static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
      self.mode = static_image_mode
      self.numHands = max_num_hands
      self.detectionConfidence = min_detection_confidence
      self.trackingConfidence = min_tracking_confidence

      self.mpDrawing = mp.solutions.drawing_utils
      self.mpHands = mp.solutions.hands
      self.hands = self.mpHands.Hands(
          self.mode, self.numHands, self.detectionConfidence, self.trackingConfidence)

  def findHands(self, image, draw=True):
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      results = self.hands.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          if draw:
            self.mpDrawing.draw_landmarks(
                image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

      return image


def main():

  cap = cv2.VideoCapture(0)
  detector = handDetector()

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = detector.findHands(image)
    cv2.imshow('Sign-Language', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

  cap.release()


if __name__ == "__main__":
    main()
