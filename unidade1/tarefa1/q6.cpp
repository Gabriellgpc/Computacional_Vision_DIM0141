//Programa que tira a media aritmetica de N imagens
//Este programa carrega todas as imagens de um arquivo, cujo caminho (path)
//esta indicado pela variavel path, e calcula a media de todas essas imagens
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char**argv)
{
  String path("./imagensComRuido/*.jpg");
  std::vector<String> fn;
  std::vector<Mat> imagens;
  Mat media;

  glob(path, fn, true);
  //carrega as N imagens
  for(size_t i = 0; i < fn.size(); i++)
  {
    Mat im = imread(fn[i], CV_LOAD_IMAGE_GRAYSCALE);
    if(!im.data)continue;//caso falhe na leitura, vai para o proximo arquivo
    imagens.push_back(im);
  }

  if(imagens.empty())
  {
    std::cerr << "Erro ao carrega as imagens de " << path << "\n";
    return -1;
  }

  media = Mat::zeros(imagens[0].size(), CV_32FC1);
  //calcula da media das N imagens
  for(size_t i = 0; i < imagens.size(); i++)
  {
    for(int y = 0; y < imagens[0].rows; y++)
    for(int x = 0; x < imagens[0].cols; x++)
    {
      media.at<float>(y,x) += imagens[i].at<uint8_t>(y,x)/(float)imagens.size();
    }
  }
  media.convertTo(media, CV_8UC1, 1.0, 0);

  imshow("Media", media);
  imwrite("Media.png",media);

  waitKey(0);
  return 0;
}
