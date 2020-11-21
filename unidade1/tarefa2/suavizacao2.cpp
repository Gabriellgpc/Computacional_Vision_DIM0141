#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define MEDIA_SIZE 10.0

void convFilter(const Mat &src, Mat &dest, const Mat &mask);

int main(int argc, char **argv)
{
  Mat img, result, tmp;
  Mat blurImg, contourImg;
  Mat mask;
  float kernel_media[(int)(MEDIA_SIZE * MEDIA_SIZE)];
  float kernel_gaussian[] = {1, 2, 1,
                             2, 4, 2,
                             1, 2, 1};
  float kernel_laplaciano[] = {1, 1, 1,
                               1, -8, 1,
                               1, 1, 1};

  for (uint i = 0; i < (MEDIA_SIZE * MEDIA_SIZE); i++)
    kernel_media[i] = 1.0 / (MEDIA_SIZE * MEDIA_SIZE);

  img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data)
  {
    std::cerr << "Erro ao carregar imagem\n";
    return -1;
  }

  //suavizacao gaussiana
  mask = Mat(3, 3, CV_32F, kernel_gaussian) / 16.0;
  convFilter(img, contourImg, mask);

  //realçar contornos
  mask = Mat(3, 3, CV_32F, kernel_laplaciano);
  convFilter(contourImg, contourImg, mask);
  contourImg.convertTo(tmp, CV_8U, 255.0 / 2040.0, 127.0);

  /*Exibir e salvar imagem*/
  imshow("Contornos", tmp);
  imwrite("contorno.png", tmp);

  //Borramento
  mask = Mat(MEDIA_SIZE, MEDIA_SIZE, CV_32F, kernel_media);
  convFilter(img, blurImg, mask);
  blurImg.convertTo(tmp, CV_8U);

  /*Exibir e salvar imagem*/
  imshow("Borrada", tmp);
  imwrite("borrada.png", tmp);

  result = blurImg - contourImg;
  result.convertTo(result, CV_8U);

  //trecho apenas para exibir e salvar as imagens
  imshow("Entrada", img);
  imshow("Saida", result);
  imwrite("entradaSuavizacao2.png", img);
  imwrite("saidaSuavizacao2.png", result);

  waitKey(0);
  return 0;
}

void convFilter(const Mat &src, Mat &dest, const Mat &mask)
{
  float sum;
  Mat src_f;

  src.convertTo(src_f, CV_32F);
  dest = Mat::zeros(src.size(), CV_32F);

  for (int y = mask.rows / 2; y < (src.rows - mask.rows / 2); y++)
    for (int x = mask.cols / 2; x < (src.cols - mask.cols / 2); x++)
    {
      sum = 0;
      for (int v = 0; v < mask.rows; v++)
        for (int u = 0; u < mask.cols; u++)
        {
          sum += src_f.at<float>(y - mask.rows / 2 + v, x - mask.cols / 2 + u) * mask.at<float>(v, u);
        }
      dest.at<float>(y, x) = sum;
    }
}
