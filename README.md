## EKF SLAM algorithm for Sensor Signal and Data Processing (EE5020)

Forked from [https://github.com/Attila94/EKF-SLAM](https://github.com/Attila94/EKF-SLAM)

![EKF SLAM Demo](https://github.com/Attila94/EKF-SLAM/blob/master/images/static_landmarks.png)

Basic algorithm creates robot object, generates (semi) random trajectory and estimates trajectory based on range-bearing measurements.
Advanced algorithm is able to generate moving landmarks, but landmark classification has not been implemented.

For more information, read his [report](https://github.com/Attila94/EKF-SLAM/blob/master/report.pdf).

Basic algorithm is based on “Robot Mapping - WS 2013/14,” 21 10 2018. [Online]. Available: http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/.

### Usage

Make sure `numpy` and `matplotlib` are installed and run `slam.py`.

The results of pictures will be in /fig/ekf.png and /fig/ekf_truth_last.png. 

Maybe this can conclude the inconsistency of EKF.
