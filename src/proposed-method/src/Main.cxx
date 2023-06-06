/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/

#include "Globals.hpp"
#include "ImageUtils.hpp"
#include "ImageSplitter.hpp"

#include "01-Preprocessing.hpp"
#include "02-Segmentation.hpp"
#include "03-AdjacentBoneSeparation.hpp"

#include "boost/tuple/tuple.hpp"
#include "boost/lexical_cast.hpp"

/**
Class for unified and platform-independent
access to filenames used in the program.
*/
struct FilenameDb {

    string m_tempDir;
    string m_inputFilename;
    string m_outputFilename;

    FilenameDb(string input, string output, string tempDir) {
        m_inputFilename = input;
        m_outputFilename = output;
        m_tempDir = tempDir;
    }

    string input()          { return m_inputFilename; }
    string output()         { return m_outputFilename; }
    string sheetness()      { return m_tempDir + "/sheetness.nii"; }
    string roi()            { return m_tempDir + "/roi.nii"; }
    string softTissueEst()  { return m_tempDir + "/soft-tissue-est.nii"; }
    string segmOutputAll()  { return m_tempDir + "/segm-output.nii"; }

    string segmOutputPart(unsigned part) {
        string filename =
            "/gc-output-part-" + boost::lexical_cast<string>(part) + ".nii";
        return m_tempDir + filename;
    }
};

int main(int argc, char * argv [])
{
	//-----------------------------------
	// Program argument parsing
	//-----------------------------------
    string argv1 = "C:\\Project\\bone-segmentation\\sample-volumes\\001-CT.nii";
    string argv2 = "C:\\Project\\bone-segmentation\\output";
    string argv3 = "C:\\Project\\bone-segmentation\\output\\output.nii";
    FilenameDb filenames(argv1, argv3, argv2);

	//-----------------------------------
	// Preprocessing
	//-----------------------------------

    float sigmaSmallScale = 1.5;
    vector<float> sigmasLargeScale;
    sigmasLargeScale.push_back(0.6);
    sigmasLargeScale.push_back(0.8);

    logSetStage("Init");
    logger("Loading image %s") % filenames.input();
    ShortImagePtr inputCT = ImageUtils<ShortImage>::readImage(filenames.input());

    logSetStage("Preprocessing");
    UCharImagePtr roi;
    FloatImagePtr sheetness;
    UCharImagePtr softTissueEst;
    boost::tie(roi, sheetness, softTissueEst) = Preprocessing::compute(inputCT, sigmaSmallScale, sigmasLargeScale);

    logSetStage("Disassembly");
    vector<ImageRegion> subRegions;
    subRegions = ImageSplitter<UCharImage>::splitIntoRegions(roi);

    // save results of the preprocessing
    // the sheetness is scaled to -100,100 and saved as char-image
    // to save memory on the disk
    ImageUtils<UCharImage>::writeImage(filenames.roi(), roi);
    ImageUtils<UCharImage>::writeImage(filenames.softTissueEst(), softTissueEst);
    ImageUtils<CharImage>::writeImage(filenames.sheetness(), FilterUtils<FloatImage,CharImage>::linearTransform(sheetness,100,0));

    

    // at this point, the images inputCT, roi, sheetness and softTissueEst
    // no longer exist and therefore they don't occupy memory.
    // this is important in case the input CT is very large (cca 1GB) and
    // the graph-cut segmentation will be performed per partes.

	//-----------------------------------
	// Segmentation
	//-----------------------------------

    // graph-cut segmentation per-partes
    for (unsigned i = 0; i < subRegions.size(); ++i) {

        logSetStage("Segmentation#" + boost::lexical_cast<string>(i+1));

        ImageRegion region = subRegions[i];

        // load the images needed for the graph-cut (however,
        // only the region of interest of each image is loaded)
        ShortImagePtr inputCT =
            ImageUtils<ShortImage>::readImage(filenames.input(),region);
        UCharImagePtr roi =
            ImageUtils<UCharImage>::readImage(filenames.roi(),region);
        UCharImagePtr softTissueEst =
            ImageUtils<UCharImage>::readImage(filenames.softTissueEst(),region);
        FloatImagePtr sheetness =
            FilterUtils<CharImage,FloatImage>::linearTransform(
                ImageUtils<CharImage>::readImage(filenames.sheetness(),region),
                0.01, 0
            );

        // segment
        UCharImagePtr gcResult = Segmentation::compute(
            inputCT, roi, sheetness, softTissueEst);

        // save the result
        logger("Saving temporal result to %s") % filenames.segmOutputPart(i);
        ImageUtils<UCharImage>::writeImage(filenames.segmOutputPart(i), gcResult);

    }


    logSetStage("Assembly");
    logger("Assembling temporal results");

    UCharImagePtr assembledResult = FilterUtils<UCharImage>::createEmptyFrom(
        ImageUtils<UCharImage>::readImage(filenames.roi()));
    for (unsigned i = 0; i < subRegions.size(); ++i) {

        UCharImagePtr partialResult =
            ImageUtils<UCharImage>::readImage(filenames.segmOutputPart(i));

        assembledResult = FilterUtils<UCharImage>::paste(
            partialResult, partialResult->GetLargestPossibleRegion(),
            assembledResult, subRegions[i].GetIndex()
        );
    }


	//-----------------------------------
	// Bone Separation
	//-----------------------------------
    logSetStage("Bone Separation");
    UCharImagePtr finalResult = BoneSeparation::compute(assembledResult);

	//-----------------------------------
	// Saving the result, Cleaning up
	//-----------------------------------

    logSetStage("Saving");
    logger("Writing the result to %s") % filenames.output();

    ImageUtils<UCharImage>::writeImage(filenames.output(), finalResult);

	return EXIT_SUCCESS;
}
