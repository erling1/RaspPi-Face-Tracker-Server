import RPi.GPIO as GPIO
import time

# --- GPIO Pin Configuration ---
servo_pin = 18  # GPIO pin 18 for PWM (make sure it's connected to your servo)

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup(servo_pin, GPIO.OUT)  # Set servo pin as an output

# --- PWM Setup ---
pwm = GPIO.PWM(servo_pin, 50)  # Set frequency to 50Hz (standard for most servos)
pwm.start(0)  # Start PWM with 0% duty cycle (servo is in neutral position)

# --- Function to Set Servo Angle ---
def set_angle(angle):
    # Calculate duty cycle (angle between 0 and 180)
    duty = float(angle) / 18 + 2
    GPIO.output(servo_pin, True)  # Send PWM signal to servo
    pwm.ChangeDutyCycle(duty)  # Change the duty cycle to set the angle
    time.sleep(1)  # Wait for the servo to reach the position
    GPIO.output(servo_pin, False)  # Stop sending PWM signal
    pwm.ChangeDutyCycle(0)  # Stop PWM

# --- Main Loop ---
try:
    while True:
        # Test servo rotation: 0° to 180° and back
        print("Rotating to 0°")
        set_angle(0)
        time.sleep(2)  # Wait for 2 seconds

        print("Rotating to 90°")
        set_angle(90)
        time.sleep(2)  # Wait for 2 seconds

        print("Rotating to 180°")
        set_angle(180)
        time.sleep(2)  # Wait for 2 seconds

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    pwm.stop()  # Stop PWM
    GPIO.cleanup()  # Clean up GPIO to free resources
