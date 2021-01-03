package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"io"
	"os"
	"path"
)

// MINST image dimension in pixels.
const (
	Width  = 28
	Height = 28
)

// MNIST database file names.
const (
	TrainingImageFileName = "train-images-idx3-ubyte.gz"
	TrainingLabelFileName = "train-labels-idx1-ubyte.gz"
	TestImageFileName     = "t10k-images-idx3-ubyte.gz"
	TestLabelFileName     = "t10k-labels-idx1-ubyte.gz"
)

// Image represents a MNIST image. It is a array a bytes representing the color.
// 0 is black (the background) and 255 is white (the digit color).
type Image [Width * Height]byte

// Label is the digit label from 0 to 9.
type Label int8

// Set represents the data set with the images paired with the labels.
type Set struct {
	Images []*Image
	Labels []Label
}

type imageFileHeader struct {
	Magic     int32
	NumImages int32
	Height    int32
	Width     int32
}

type labelFileHeader struct {
	Magic     int32
	NumLabels int32
}

// Magic keys are used to check file formats.
const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
)

// readImage reads a image from the file and returns it.
func readImage(r io.Reader) (*Image, error) {
	img := &Image{}
	err := binary.Read(r, binary.BigEndian, img)
	return img, err
}

// LoadImageFile opens the image file, parses it, and returns the data in order.
func LoadImageFile(name string) ([]*Image, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}

	header := imageFileHeader{}

	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != imageMagic ||
		header.Width != Width ||
		header.Height != header.Height {
		return nil, errors.New("mnist: invalid format")
	}

	images := make([]*Image, header.NumImages)
	for i := int32(0); i < header.NumImages; i++ {
		images[i], err = readImage(reader)
		if err != nil {
			return nil, err
		}
	}

	return images, nil
}

// LoadLabelFile opens the label file, parses it, and returns the labels in
// order.
func LoadLabelFile(name string) ([]Label, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}

	header := labelFileHeader{}

	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != labelMagic {
		return nil, err
	}

	labels := make([]Label, header.NumLabels)
	for i := int32(0); i < header.NumLabels; i++ {
		err = binary.Read(reader, binary.BigEndian, &labels[i])
		if err != nil {
			return nil, err
		}
	}

	return labels, nil
}

// ColorModel implements the image.Image interface.
func (img *Image) ColorModel() color.Model {
	return color.GrayModel
}

// Bounds implements the image.Image interface.
func (img *Image) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{0, 0},
		Max: image.Point{Width, Height},
	}
}

// At implements the image.Image interface.
func (img *Image) At(x, y int) color.Color {
	return color.Gray{Y: img[y*Width+x]}
}

// Set modifies the pixel at (x,y).
func (img *Image) Set(x, y int, v byte) {
	img[y*Width+x] = v
}

// Get returns the i-th image and its label.
func (s *Set) Get(i int) (*Image, Label) {
	return s.Images[i], s.Labels[i]
}

// Load loads the whole MINST database and returns the training set and the test
// set.
func Load(dir string) (training, test *Set, err error) {
	trainingImages, err := LoadImageFile(path.Join(dir, TrainingImageFileName))
	if err != nil {
		return nil, nil, err
	}

	trainingLabels, err := LoadLabelFile(path.Join(dir, TrainingLabelFileName))
	if err != nil {
		return nil, nil, err
	}

	if len(trainingImages) != len(trainingLabels) {
		return nil, nil, errors.New("mnist: training size mismatch")
	}

	training = &Set{
		Images: trainingImages,
		Labels: trainingLabels,
	}

	testImages, err := LoadImageFile(path.Join(dir, TestImageFileName))
	if err != nil {
		return nil, nil, err
	}

	testLabels, err := LoadLabelFile(path.Join(dir, TestLabelFileName))
	if err != nil {
		return nil, nil, err
	}

	if len(testImages) != len(testLabels) {
		return nil, nil, errors.New("mnist: test size mismatch")
	}

	test = &Set{
		Images: testImages,
		Labels: testLabels,
	}

	return
}
