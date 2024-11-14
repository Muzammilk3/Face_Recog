import cv2
import face_recognition
import csv
import os
import numpy as np

# Function to capture an image of a student
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        cap.release()
        return None

    cv2.imshow("Capture", frame)
    cv2.waitKey(1000)  # Show the captured frame for a second
    cap.release()
    cv2.destroyAllWindows()

    return frame  # Return the captured frame

# Function to register a new student
def register_student(name, branch):
    # Capture the image
    frame = capture_image()
    if frame is None:
        print("Image capture failed, please try again.")
        return

    # Encode the face
    face_encodings = face_recognition.face_encodings(frame)
    if not face_encodings:
        print("No face detected, please try again.")
        return
    
    encoding = face_encodings[0]

    # Register the student
    with open('students.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, branch, encoding.tolist()])  # Convert numpy array to list for CSV
    print(f"{name} registered successfully!")

# Function to verify a student's identity
def verify_student():
    if not os.path.exists('students.csv'):
        print("No registered students found.")
        return

    known_face_encodings = []
    known_face_names = []

    # Load known faces from CSV
    with open('students.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            known_face_names.append(row[0])
            known_face_encodings.append(np.array(eval(row[2])))  # Convert string back to numpy array

    # Capture image for verification
    frame = capture_image()
    if frame is None:
        print("Image capture failed, please try again.")
        return

    # Encode the captured face
    face_encodings = face_recognition.face_encodings(frame)
    if not face_encodings:
        print("No face detected, please try again.")
        return

    # Compare the captured face with known faces
    face_to_verify = face_encodings[0]
    
    # Find the closest match
    distances = face_recognition.face_distance(known_face_encodings, face_to_verify)
    best_match_index = np.argmin(distances)  # Get the index of the closest match
    best_match_distance = distances[best_match_index]  # Distance of the closest match
    
    # Set a threshold to determine if a match is valid
    threshold = 0.6  # You can adjust this value based on testing
    if best_match_distance < threshold:
        name = known_face_names[best_match_index]
        print(f"Face recognized: {name}")
    else:
        print("Face not recognized.")

# Main function to control the flow of the program
def main():
    while True:
        choice = input("Enter 'r' to register a student, or 'v' to verify a student, or 'q' to quit: ")
        if choice == 'r':
            name = input("Enter student's name: ")
            branch = input("Enter student's branch: ")
            register_student(name, branch)
        elif choice == 'v':
            verify_student()
        elif choice == 'q':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
