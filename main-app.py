import sys
import argparse
import cv2
from matplotlib import pyplot as plt
from Libraries.ExtractKeypoints.ExtractKeypoints import extractKeypoints


def main(args):
    print('----|| INIT MODULE ||----')
    img1 = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = extractKeypoints(img1)
    img2 = cv2.imread(args.ref_image, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = extractKeypoints(img2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)

    img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
    img5 = cv2.drawKeypoints(img2, kp2, outImage=None)

    if args.visual:
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img4)
        axarr[1].imshow(img5)
        plt.show()

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
        plt.imshow(img3)
        plt.show()

    score = 0
    for match in matches:
        score += match.distance
    if score / len(matches) < args.thres:
        print("RESULT: Signature does match with score = {}".format(100-(score / len(matches))))
    else:
        print("RESULT: Signature match with score."
    print('----|| MODULE ENDED ||----')


def setup(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Provide signature image path needs to be verified', type=str, default='Data/signature.jpg')
    parser.add_argument('--ref_image', help='Provide reference signature image path needs to be verified', type=str, default='Data/ref-signature.jpg')
    parser.add_argument('--thres', help='Signature matching threshold', type=int, default=20)
    parser.add_argument('--visual', help='Visulisation of result', type=bool, default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    '''
    Configuration
    '''
    arguments = setup(sys.argv[1:])
    main(args=arguments)
