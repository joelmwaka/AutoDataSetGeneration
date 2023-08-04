# Automated Dataset Generator

## Proposed Procedure

1. Manually label a single image of the object whose dataset needs to be generated.

    In our case, we intend to detect H-Tafel images from images captured by a camera.
    We have a single image of how the H-Tafel looks like and will manually select the keypoints that we want our model to later on extract from this particular image.

    The key points are saved in a <...> file.

    The H-Tafel has a large capital H on it with a white background. The key points we save are the vertices of the letter 'H' in the original object image. Each of these points is then saved as a pixel coordinate with its index. The indices are assigned according to which vertice the point represents.

    1___2         5___6
    |   |         |   |
    |   |         |   |
    |   |3_______4|   |
    |                 |
    |   10________9   |
    |   |         |   |
    |   |         |   |
    12__11        8___7

2. Alter the orientation of the annotated image and perform other altering functions on it like adding some random noise, 
