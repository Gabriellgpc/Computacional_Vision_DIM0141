#include <string>
#include <opencv2/opencv.hpp>
#include <list>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

//180 orientacoes diferentes entre [0,pi)
static const uint32_t ang_resolution = 180;
static const long double ang_step = M_PI / ang_resolution;
static const uint32_t min_square_size = 5; //menor lado considerado de um quadrado [px]

class MySquare
{
public:
    MySquare() : _xc(0), _yc(0), _size(0), _theta(0) {}
    MySquare(uint32_t xc, uint32_t yc, double theta, uint32_t size) : _xc(xc), _yc(yc), _size(size), _theta(theta) {}
    inline Point2f center()const{ return Point2f((float)_xc, (float)_yc); }
    inline uint32_t &xc() { return _xc; }
    inline uint32_t &yc() { return _yc; }
    inline uint32_t &size() { return _size; }
    inline double &theta() { return _theta; }
    inline uint32_t isize()const{return _size - min_square_size;}
    inline uint32_t itheta() //devolve o inteiro correspondente ao angulo no intervalo [0,90)
    {
        if (_theta < M_PI_2f64)
            _theta += 2.0 * M_PI;
        return uint32_t(round(_theta * (ang_resolution / M_PI))) % ang_resolution;
    }

private:
    uint32_t _xc, _yc;
    uint32_t _size;
    double _theta; //angulo com relacao ao eixo horizontal
};

//img: imagem com fundo branco e quadrados pretos
//max_detection: -1 para tentar achar todos, ou um número inteiro para limitar
std::list<MySquare> myHoughtTransform4Squares(const Mat &img, const int &max_detection = -1);
void _drawSquare(const Mat &src, Mat &dest, const MySquare square);
void drawSquares(const Mat &src, Mat &dest, const std::list<MySquare> &squares);
void convFilter(const Mat &src, Mat &dest, const Mat &mask);

int main(int argc, char **argv)
{
    Mat img, output;
    std::list<MySquare> squares;

    /*
    img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (!img.data)
    {
        std::cerr << "Erro ao carregar imagem\n";
        return -1;
    }
    */
    img = Mat::ones(Size(400,400), CV_8U)*255;
    std::cout << "Looking for black squares...\n";

    uint32_t xc, yc;
    const uint32_t r = 100;
    for(float ang = 0; ang < 2.0*M_PI; ang += 2.0*M_PI/100)
    {   
        xc = 200 +((float)r)*cos(ang);
        yc = 200 + ((float)r)*sin(ang);
        squares.push_back( MySquare(xc, yc, ang, 10));
    }

    drawSquares(img, output, squares);

    std::cout << "Quantidade de quadrados detectados identificados:" << squares.size() << '\n';
    //trecho apenas para exibir e salvar as imagens
    imshow("Entrada", img);
    imshow("Saida", output);

    //aguarda uma tecla para encerrar o programa
    waitKey(0);
    return 0;
}

std::list<MySquare> myHoughtTransform4Squares(const Mat &img, const int &max_detection)
{
}

void _drawSquare(const Mat &src, Mat &dest, MySquare square)
{   
    RotatedRect rSquare = RotatedRect(square.center(), Size2f(square.size(),square.size()), square.theta());
    Point2f vertices[4];
    rSquare.points(vertices);
    for(uint8_t i = 0; i < 4; i++) 
        line(dest, vertices[i], vertices[(i+1)%4], Scalar(255,0,0));
}
void drawSquares(const Mat &src, Mat &dest, const std::list<MySquare> &squares)
{   
    src.copyTo(dest);
    cvtColor(dest,dest,COLOR_GRAY2RGB);
    
    for(auto sq_it = squares.begin(); sq_it != squares.end(); sq_it++)
        _drawSquare(dest, dest, *sq_it);
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
