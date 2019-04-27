# Signature Verification

Once the signature is localized we will perform cropping on the whole document to extract the signature from the document. Then we are going to recognize two signature using image processing without CNN. The process involves taking two signature of the same individual.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
sudo pip install numpy
sudo pip install scipy
sudo pip install matplotlib
sudo pip install opencv-python
sudo pip install skimage
```

## Usage

```bash
python main.py <first-signature-path> <second-signature-path>
```

## Technical Details
```text
Process:

In this project, we are going to recognize two signature using image processing without CNN.

The process involves taking two signature of the same individual
Before starting the whole process we will use Contrast-limited adaptive histogram equalization (CLAHE) to properly equalize the histogram.

Read that image into grayscale.
After that, we have to find and segment out the strokes in the signature but before that, we will normalize the image.

As the segmentation of strokes has been done, we have to find the orientation of each and every pixel of the normalized image.

Now we will estimate the signature stroke across the signature image. It is done by calculating the frequency of strokes for each chunk of the image ie. distributing the signature image into a fixed block size.

So in the final, we will enhance the image by using oriented filters. Here we exploited the concept of Gabor filters using numpy libraries.

Then we will do the proper thresholding and normalize the image between 0 and 1. 

After this process, we try to thinning the thresholded signature image and removing irrelevant points and noise in the image.

We will try to extract features like corners and curves out of the image.

After extracting the features we will use ORB (Oriented FAST and Rotated BRIEF) to get the descriptors because of their several advantages over SIFT and SURF here.

In the final, we will use any feature matching algorithm. Here in this project, we are using brute-force matcher algorithm.

After this, we define the signature is matched or not by giving it proper thresholding depending upon the input image(if we put constraints on input image this threshold can be generalized).
```

## Flow Diagram
![Data - flow](https://github.com/ashutoshraj/SigantureRecognition/blob/master/Data/flow_diagram.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GPL](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
