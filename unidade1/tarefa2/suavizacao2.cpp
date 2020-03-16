#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define T1 30

int main(int argc, char** argv)
{
  Mat img;
  Mat blurImg, blurImg2;
  Mat res;
  Mat cannyImg;

  img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if(!img.data)
  {
    std::cerr << "Erro ao carregar imagem\n";
    return -1;
  }

  //Aplica filtro de suavização, filtro de borramento
  blur(img, blurImg, Size(5,5));
  blur(img, blurImg2, Size(3,3));
  //borramento
  res = blurImg.clone();
  //Canny
  Canny(img, cannyImg, T1, 3*T1);
  //recupera bordas
  for(int y = 0; y < img.rows; y++)
  for(int x = 0; x < img.cols; x++)
  {
    if(cannyImg.at<uint8_t>(y,x) != 0)
      res.at<uint8_t>(y,x) = blurImg2.at<uint8_t>(y,x);
  }

  imshow("Entrada", img);
  imshow("Canny", cannyImg);
  imshow("Suavizada", blurImg);
  imshow("Suavizada2", blurImg2);
  imshow("Suavização seletiva", res);
  waitKey(0);
  return 0;
}
