import csv
import pytest
import cv2
import vision
from helper_types import BoundingBox, ExpectedGamePiece


def read_test_data_csv(fname: str):
    with open(fname) as f:
        result = []
        for image, cone_present, cube_present, x1, y1, x2, y2 in csv.reader(f):
            result.append(
                (
                    image,
                    cone_present == "True",
                    cube_present == "True",
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
            )
    return result


images = read_test_data_csv("test/expected.csv")


@pytest.mark.parametrize("filename,cone_present,cube_present,x1,y1,x2,y2", images)
def test_sample_images(
    filename: str,
    cone_present: bool,
    cube_present: bool,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
):
    image = cv2.imread(f"./test/{filename}")
    assert image is not None
    bounding_box = BoundingBox(x1, y1, x2, y2)
    if cone_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CONE)
            == True), "Cone present in image but not found by detector"
        
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CUBE)
            == False), "Cone present in image but detector found cube"
        
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.BOTH)
            == True), "Cone present in image but detector found neither"
        

    if cube_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CUBE)
            == True), "Cube present in image but not found by detector"
        
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CONE)
            == False), "Cube present in image but detector found cube"
        
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.BOTH)
            == True), "Cube present in image but detector found neither"
        
    if not cube_present and not cone_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CUBE)
            == False), "Nothing present in image but detector found cube"
        
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CONE)
            == False), "Nothing present in image but detector found cone"
        
