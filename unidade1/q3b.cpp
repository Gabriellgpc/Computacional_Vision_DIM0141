//Este programa recebe uma imagem como entrada e inverte a mesma, espelha na vertical,
//exibi e salva o resultado
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char**argv)
{
  Mat img;
  Mat imgFlip;

  img = imread(argv[1],CV_LOAD_IMAGE_COLOR);

  if(!img.data){
    std::cout << "imagem nao carregou corretamente\n";
    return(-1);
  }
  imshow("Original", img);

  imgFlip = img.clone();

  for(int y = 0; y < img.rows; y++)
  for(int x = 0; x < img.cols; x++)
  {
    imgFlip.at<Vec3b>(y,img.cols - 1 - x) = img.at<Vec3b>(y,x);
  }

  imshow("Imagem Invertida Horizontalmente", imgFlip);
  imwrite("entrada_q3b.png", img);
  imwrite("flipHorizontal.png", imgFlip);

  waitKey(0);
  return 0;
}
