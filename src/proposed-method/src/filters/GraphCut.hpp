/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/
#pragma once

#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNeighborhoodIterator.h"
#include <vector>
#include <algorithm>
#include "ImageUtils.hpp"
#include "graph.h"

template<unsigned int Dimension> // dimension of the input image
class GraphCutSegmentation {

public:
    typedef unsigned LabelID;
    typedef int PixelID;
    typedef int EnergyTerm;

    typedef typename itk::Image<LabelID, Dimension> LabelIdImage;
    typedef typename itk::Image<PixelID, Dimension> PixelIdImage;
    typedef typename LabelIdImage::IndexType ImageIndex;
    typedef typename LabelIdImage::SizeType ImageRegionSize;
    typedef typename LabelIdImage::Pointer LabelIdImagePointer;
    typedef typename PixelIdImage::Pointer PixelIdImagePointer;

    typedef short EdgeCapacityType;
    typedef long long FlowType;

    typedef Graph<EdgeCapacityType,EdgeCapacityType,FlowType> GraphType;

    struct DataCostFunction {
        /* Compute data cost of assigning @label to the pixel with index @idx */
        virtual int compute(ImageIndex idx, LabelID label) = 0;
    };

    struct SmoothnessCostFunction {
        /* Compute smooth cost between pixels at positions @idx1 and @idx2 */
        virtual int compute(ImageIndex idx1, ImageIndex idx2 ) = 0;
    };

private:
    PixelIdImagePointer _pixelIdImage;
    /* Image to hold a label for each pixel */
    LabelIdImagePointer _labelIdImage;
    /* Third-party graph-cut algorithm*/
    GraphType *_gc;
    /* Number of pixels in ROI */
    unsigned int _totalPixelsInROI;
    /* Number of neighbors */
    unsigned int _totalNeighbors;
    //============================
    // Member functions:

    void assignIdsToPixels(LabelIdImagePointer  img) {
        // create a new image
        _pixelIdImage = ImageUtils<PixelIdImage>::createEmpty(img->GetLargestPossibleRegion().GetSize());

        // fill it with identifiers
        itk::ImageRegionIteratorWithIndex<LabelIdImage> it(
            img, img->GetLargestPossibleRegion());

        _totalPixelsInROI = 0;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

            LabelID label = it.Get();

            // pixels outside ROI are assigned -1
            if (label == 0) {
                _pixelIdImage->SetPixel(it.GetIndex(), -1 );
                continue;
            }

            // pixel id
            _pixelIdImage->SetPixel(it.GetIndex(), _totalPixelsInROI);

            _totalPixelsInROI++;
        }

    }

    void initializeDataCosts(DataCostFunction * dataCostFunction) {

        itk::ImageRegionIteratorWithIndex<PixelIdImage> itID(_pixelIdImage, _pixelIdImage->GetLargestPossibleRegion());

        for (itID.GoToBegin(); !itID.IsAtEnd(); ++itID) {
            PixelID pixelId = itID.Get();
            if (pixelId >= 0) {
                ImageIndex pixelIndex = itID.GetIndex();
                int dataCostSource = dataCostFunction->compute(pixelIndex, 0);
                int dataCostSink = dataCostFunction->compute(pixelIndex, 1);
                _gc->add_tweights(pixelId, dataCostSource, dataCostSink);

            }
        } // iteration through image
    }

    void initializeNeighbours(
        SmoothnessCostFunction * smoothnessCostFunction
    ) {

        itk::ImageRegionIteratorWithIndex<PixelIdImage>
            it(_pixelIdImage, _pixelIdImage->GetLargestPossibleRegion());

        ImageRegionSize imageSize = _pixelIdImage->GetLargestPossibleRegion().GetSize();

        _totalNeighbors = 0;

        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

            ImageIndex centerPixelIndex = it.GetIndex();
            PixelID centerPixelID = _pixelIdImage->GetPixel(centerPixelIndex);

            // skip regions outside ROI
            if (centerPixelID < 0)
                continue;

            // examine forward neighbours in all directions
            for (unsigned  dim=0; dim < Dimension; ++dim)
                for (unsigned offset = 1; offset <=1; ++offset) {

                    int newCoord = centerPixelIndex[dim] + offset;
                    bool insideImage = (newCoord >= 0) && (newCoord < (int)imageSize[dim]);

                    if (insideImage) {

                        ImageIndex neighIndex = centerPixelIndex;
                        neighIndex[dim] = newCoord;

                        PixelID neighPixelID = _pixelIdImage->GetPixel(neighIndex);

                        if (neighPixelID >= 0) {

                            assert(neighPixelID > centerPixelID);

                            int smoothCostFromCenter = smoothnessCostFunction->compute(
                                    centerPixelIndex, neighIndex);
                            int smoothCostToCenter = smoothnessCostFunction->compute(
                                    neighIndex, centerPixelIndex);

                            _gc->add_edge(centerPixelID, neighPixelID,
                                smoothCostFromCenter,smoothCostToCenter);

                            _totalNeighbors++;
                        }
                    } // if inside

                } // for dim

        } // iterating through ROI image
    }

    void updateLabelImageAccordingToGraph() {

        // update the resulting (labelled) image
        itk::ImageRegionIteratorWithIndex<PixelIdImage> it(
            _pixelIdImage, _pixelIdImage->GetLargestPossibleRegion());

        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

            PixelID pixelId = it.Get();

            // skip pixels outside ROI
            if (pixelId == -1) {
                it.Set(0);
                continue;
            }

            // update labels
            LabelID newLabel;
            newLabel = (_gc->what_segment(pixelId) == GraphType::SOURCE) ? 1 : 0;
            _labelIdImage->SetPixel(it.GetIndex(), newLabel);
        }
    }

    void buildGraph(LabelIdImagePointer labelImage,DataCostFunction * dataCostFunction,SmoothnessCostFunction * smoothnessCostFunction) {
        assignIdsToPixels(labelImage);

        logger("Building graph, %d nodes") % _totalPixelsInROI;

        _gc = new GraphType(_totalPixelsInROI, 3 * _totalPixelsInROI);
        _gc->add_node(_totalPixelsInROI);

        initializeDataCosts(dataCostFunction);

        initializeNeighbours(smoothnessCostFunction);
        logger("%d t-links added") % _totalNeighbors;

        _labelIdImage = labelImage;
    }
    /*
        Perform alpha-expansions and return the labeled image
    */
    LabelIdImagePointer compute() {
        assert(_gc != NULL);
        logger("Graph built. Computing the max flow");
        _gc->maxflow();
        logger("Max flow computed");
        updateLabelImageAccordingToGraph();
        // Ende :)
        delete _gc;
        return _labelIdImage;
    }

public:
    // Constructor
    GraphCutSegmentation(): _gc(NULL)
    { /* empty body */ };

    LabelIdImagePointer optimize(LabelIdImagePointer roiImage, DataCostFunction * dataCostFunction, SmoothnessCostFunction * smoothnessCostFunction) {
        //ProcessInfo::printStatus("Going to build graph");
        buildGraph(roiImage, dataCostFunction, smoothnessCostFunction);
        //ProcessInfo::printStatus("Graph built");
        return compute();
    }
};