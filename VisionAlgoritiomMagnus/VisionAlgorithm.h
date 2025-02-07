#include "pch.h"
// MatchToolDlg.h: 標頭檔
//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include <tchar.h>
#include <fstream>

using namespace cv;
using namespace std;
struct s_TemplData
{
	vector<Mat> vecPyramid;
	vector<Scalar> vecTemplMean;
	vector<double> vecTemplNorm;
	vector<double> vecInvArea;
	vector<BOOL> vecResultEqual1;
	BOOL bIsPatternLearned;
	int iBorderColor;
	void clear()
	{
		vector<Mat>().swap(vecPyramid);
		vector<double>().swap(vecTemplNorm);
		vector<double>().swap(vecInvArea);
		vector<Scalar>().swap(vecTemplMean);
		vector<BOOL>().swap(vecResultEqual1);
	}
	void resize(int iSize)
	{
		vecTemplMean.resize(iSize);
		vecTemplNorm.resize(iSize, 0);
		vecInvArea.resize(iSize, 1);
		vecResultEqual1.resize(iSize, FALSE);
	}
	s_TemplData()
	{
		bIsPatternLearned = FALSE;
	}
};
struct s_MatchParameter
{
	Point2d pt;
	double dMatchScore;
	double dMatchAngle;
	//Mat matRotatedSrc;
	Rect rectRoi;
	double dAngleStart;
	double dAngleEnd;
	RotatedRect rectR;
	Rect rectBounding;
	BOOL bDelete;

	double vecResult[3][3];//for subpixel
	int iMaxScoreIndex;//for subpixel
	BOOL bPosOnBorder;
	Point2d ptSubPixel;
	double dNewAngle;

	s_MatchParameter(Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
	{
		pt = ptMinMax;
		dMatchScore = dScore;
		dMatchAngle = dAngle;

		bDelete = FALSE;
		dNewAngle = 0.0;

		bPosOnBorder = FALSE;
	}
	s_MatchParameter()
	{
		double dMatchScore = 0;
		double dMatchAngle = 0;
	}
	~s_MatchParameter()
	{

	}
};
struct s_SingleTargetMatch
{
	Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
	double dMatchedAngle;
	double dMatchScore;
};
struct s_BlockMax
{
	struct Block
	{
		Rect rect;
		double dMax;
		Point ptMaxLoc;
		Block()
		{}
		Block(Rect rect_, double dMax_, Point ptMaxLoc_)
		{
			rect = rect_;
			dMax = dMax_;
			ptMaxLoc = ptMaxLoc_;
		}
	};
	s_BlockMax()
	{}
	vector<Block> vecBlock;
	Mat matSrc;
	s_BlockMax(Mat matSrc_, Size sizeTemplate)
	{
		matSrc = matSrc_;
		//將matSrc 拆成數個block，分別計算最大值
		int iBlockW = sizeTemplate.width * 2;
		int iBlockH = sizeTemplate.height * 2;

		int iCol = matSrc.cols / iBlockW;
		BOOL bHResidue = matSrc.cols % iBlockW != 0;

		int iRow = matSrc.rows / iBlockH;
		BOOL bVResidue = matSrc.rows % iBlockH != 0;

		if (iCol == 0 || iRow == 0)
		{
			vecBlock.clear();
			return;
		}

		vecBlock.resize(iCol * iRow);
		int iCount = 0;
		for (int y = 0; y < iRow; y++)
		{
			for (int x = 0; x < iCol; x++)
			{
				Rect rectBlock(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
				vecBlock[iCount].rect = rectBlock;
				minMaxLoc(matSrc(rectBlock), 0, &vecBlock[iCount].dMax, 0, &vecBlock[iCount].ptMaxLoc);
				vecBlock[iCount].ptMaxLoc += rectBlock.tl();
				iCount++;
			}
		}
		if (bHResidue && bVResidue)
		{
			Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
			Block blockRight;
			blockRight.rect = rectRight;
			minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
			blockRight.ptMaxLoc += rectRight.tl();
			vecBlock.push_back(blockRight);

			Rect rectBottom(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
			Block blockBottom;
			blockBottom.rect = rectBottom;
			minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
			blockBottom.ptMaxLoc += rectBottom.tl();
			vecBlock.push_back(blockBottom);
		}
		else if (bHResidue)
		{
			Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
			Block blockRight;
			blockRight.rect = rectRight;
			minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
			blockRight.ptMaxLoc += rectRight.tl();
			vecBlock.push_back(blockRight);
		}
		else
		{
			Rect rectBottom(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
			Block blockBottom;
			blockBottom.rect = rectBottom;
			minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
			blockBottom.ptMaxLoc += rectBottom.tl();
			vecBlock.push_back(blockBottom);
		}
	}
	void UpdateMax(Rect rectIgnore)
	{
		if (vecBlock.size() == 0)
			return;
		//找出所有跟rectIgnore交集的block
		int iSize = vecBlock.size();
		for (int i = 0; i < iSize; i++)
		{
			Rect rectIntersec = rectIgnore & vecBlock[i].rect;
			//無交集
			if (rectIntersec.width == 0 && rectIntersec.height == 0)
				continue;
			//有交集，更新極值和極值位置
			minMaxLoc(matSrc(vecBlock[i].rect), 0, &vecBlock[i].dMax, 0, &vecBlock[i].ptMaxLoc);
			vecBlock[i].ptMaxLoc += vecBlock[i].rect.tl();
		}
	}
	void GetMaxValueLoc(double& dMax, Point& ptMaxLoc)
	{
		int iSize = vecBlock.size();
		if (iSize == 0)
		{
			minMaxLoc(matSrc, 0, &dMax, 0, &ptMaxLoc);
			return;
		}
		//從block中找最大值
		int iIndex = 0;
		dMax = vecBlock[0].dMax;
		for (int i = 1; i < iSize; i++)
		{
			if (vecBlock[i].dMax >= dMax)
			{
				iIndex = i;
				dMax = vecBlock[i].dMax;
			}
		}
		ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
	}
};

class VisionAlgorithm
{
public:
	cv::Mat m_matSrc;
	cv::Mat m_matDst;
	//double m_dSrcScale;
	//double m_dDstScale;

	s_TemplData m_TemplData; // Template data toàn cục
	void LearnPattern(const int& m_iMinReduceArea);
	vector<s_SingleTargetMatch> Match(const cv::Mat& m_matSrc, cv::Mat m_matDst, s_TemplData m_TemplData, const int& m_iMinReduceArea, const bool& m_ckBitwiseNot, const bool& m_bToleranceRange, const double& m_dTolerance1, const double& m_dTolerance2, const double& m_dTolerance3, const double& m_dTolerance4, const double& m_dToleranceAngle, const double& m_dScore, const int& m_iMaxPos, const double& m_dMaxOverlap, const bool& m_bDebugMode, const bool& m_bSubPixel, const bool& m_bStopLayer1);
private:

	cv::Mat LoadImageFromFile(const std::wstring& filePath);
	int GetTopLayer(Mat* matTempl, int iMinDstLength);
	bool GetExeDir(_TCHAR* psz);
	bool comparePosWithY(const pair<Point2d, char>& lhs, const pair<Point2d, char>& rhs);
	bool comparePosWithX(const pair<Point2d, char>& lhs, const pair<Point2d, char>& rhs);
	BOOL SubPixEsimation(vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex);
	void OutputRoi(s_SingleTargetMatch sstm, const cv::Mat& m_matSrc, bool m_ckOutputROI = false);
	void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD, bool m_ckSIMD = true);
	void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI);
	void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);
	Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle);
	Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle);
	void FilterWithScore(vector<s_MatchParameter>* vec, double dScore);
	void FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap);
	Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap);
	Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax);
	void SortPtWithCenter(vector<Point2f>& vecSort);
};

