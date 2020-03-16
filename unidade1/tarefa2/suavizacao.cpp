#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  Mat img;
  Mat res;
  Mat mask;
  float kernel_gaussian[] = {1, 1, 1,
                             1, 1, 1,
                             1, 1, 1};

  img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  if(!img.data)
  {
    std::cerr << "Erro ao carregar imagem\n";
    return -1;
  }

  mask = Mat(3,3, CV_32F, kernel_gaussian)/9.0;
  //Aplica filtro de suavização, filtro de Media
  filter2D(img, res, -1, mask);

  imshow("Entrada", img);
  int key;
  string fileName;
  int count = 0;
  while(true)
  {
    //Salva cada imagem resultante para gerar o gif, com o comando convert -delay 150 tmp/* animation.gif
    fileName = string("tmp/a") + to_string(count++) + ".png";
    imwrite(fileName, res);

    filter2D(res, res, -1, mask);
    imshow("Saida", res);
    key = waitKey(0);
    if(key == 27)
      break;
  }
  imwrite("entradaSuavizacao.png",img);
  imwrite("resultadoSuavizacao.png",res);
  return 0;
}
