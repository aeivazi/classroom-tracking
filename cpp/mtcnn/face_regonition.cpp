// Regonizes faces on the input video.
// The code first splits video into frames, then find faces and five landmarks (eyes, nose tip and mouth corners) for every frame. 
// The output is written as an xml.
// Futher face tracking is done by mean of python code.
// Author Anna Eivazi
//
// The face recognition is fully base on the CascadeCNN imlementation, the citations should be given in the publictions.
//
// This file is the main function of CascadeCNN.
// A C++ re-implementation for the paper 
// Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499-1503, 2016. 
//
// Code exploited by Feng Wang <feng.wff@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD lisence.
//
// Please cite Zhang's paper in your publications if this code helps your research.

#include <experimental/filesystem>

#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>

#include <opencv2/opencv.hpp>

#include "TestFaceDetection.inc.h"

namespace pt = boost::property_tree;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace std;

using namespace FaceInception;
caffe::CaffeBinding* kCaffeBinding;
const double MIN_FACE_SIZE = 24;

pt::ptree process_image(fs::path image_path, CascadeCNN& cascade) {

	pt::ptree image_info;
	image_info.put("<xmlattr>.file", image_path.filename().string());

	Mat image = imread(image_path.string());

	vector<vector<Point2d>> landmarks;
	vector<pair<Rect2d, float>> face_boxes = cascade.GetDetection(image, 12.0 / MIN_FACE_SIZE, 0.7, true, 0.7, true, landmarks);

    //if there are different number of landmarks and boxes, something happens strange during the calculation,
	//it is better not to write anything for such image
	if (landmarks.size() != face_boxes.size()) return image_info;

	for (int i = 0; i < face_boxes.size(); i++) {

		Rect2d this_face_box = face_boxes[i].first;
		
		pt::ptree box_xml;
		box_xml.put("<xmlattr>.left", (int) this_face_box.tl().x);
		box_xml.put("<xmlattr>.top", (int) this_face_box.tl().y);
		box_xml.put("<xmlattr>.width", (int) this_face_box.width);
		box_xml.put("<xmlattr>.height", (int) this_face_box.height);

		//add label of the participant
		box_xml.put("label", boost::format("participant%d")%(i+1));
				
		for (int p = 0; p < 5; p++) {
			pt::ptree landmarks_xml;
			landmarks_xml.put("<xmlattr>.idx", p + 1);
			landmarks_xml.put("<xmlattr>.x", landmarks[i][p].x);
			landmarks_xml.put("<xmlattr>.y", landmarks[i][p].y);
			box_xml.add_child("landmarks.point", landmarks_xml);
		}

		image_info.add_child("box", box_xml);
	}

	return image_info;
}

void run_face_detection(fs::path input_path, fs::path output_xml_path, std::string dataset_name, std::string dataset_comment)
{
	string model_folder = "model\\";
	CascadeCNN cascade(model_folder + "det1-memory.prototxt", model_folder + "det1.caffemodel",
		model_folder + "det1-memory-stitch.prototxt", model_folder + "det1.caffemodel",
		model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
		model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
		model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
		-1);

	//The tree will be build during image processing
	pt::ptree tree;
	tree.put("dataset.name", dataset_name);
	tree.put("dataset.comment", dataset_comment);

	if (!fs::is_directory(input_path)) throw std::exception("Input path is not directory.");
	
	fs::path input_path_dir(input_path);
    fs::directory_iterator it(input_path_dir), eod;
    BOOST_FOREACH(fs::path const &image, std::make_pair(it, eod))
	{
		if (fs::is_regular_file(image))
		{
			cout << "Process image: " << image << '\n';
			pt::ptree image_xml = process_image(image, cascade);
			tree.add_child("dataset.images.image", image_xml);
		}
	}

	pt::xml_writer_settings<std::string> settings('\t', 1);
	pt::write_xml(output_xml_path.string(), tree, std::locale(), settings);
}

fs::path split_video_to_frames(fs::path input_video_path, int number_frames = -1)
{

	std::string video_name = input_video_path.stem().string();
	fs::path frames_dir_path = input_video_path.parent_path() / video_name;

	if (!fs::exists(frames_dir_path))
	{
		fs::create_directory(frames_dir_path);
	}

	cv::VideoCapture cap(input_video_path.string().c_str());
	cv::Mat frame;

	if (!cap.isOpened())
	{
		cout << "Error can't open video file" << endl;
		return frames_dir_path;
	}
	
	int frame_num = 0;
	while (1) {
		frame_num++;

		if (!cap.read(frame)) break;

		//stop processing if user defined limit is reached
		if (frame_num == number_frames + 1) break;

		std::string frame_filename = boost::str(boost::format("image-%s-%06d.jpg") % video_name % frame_num);
		fs::path frame_path = frames_dir_path / frame_filename;
		
		cv::imwrite(frame_path.string().c_str(), frame);

		cout << "Frame written to: " << frame_path << endl;
	}

	return frames_dir_path;
}

int process_argument(int argc, char* argv[], std::string& input_path, std::string& output_xml_path,std::string& dataset_name, std::string& dataset_comment, int& num_frames)
{
	po::options_description desc("Program Usage");

	try {

		desc.add_options()
			("help", "produce help message")
			("input,i", po::value<string>(&input_path)->required(), "path to input video or folder with frames as images")
			("output-xml,o", po::value<string>(&output_xml_path)->required(), "path to output xml")
			("dataset-name,n", po::value<string>(&dataset_name)->required(), "set dataset name")
			("dataset-comment,c", po::value<string>(&dataset_comment)->required(), "set dataset comment")
			("frames-number-limit,f", po::value<int>(&num_frames)->default_value(-1), "number of frames to process, if not defined, whole video will be processed")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);

		if (vm.count("help"))
		{
			cout << desc << endl;
			return 1;
		}

		po::notify(vm);

	}
	catch (po::error& e)
	{
		cout << e.what() << endl;
		cout << desc << endl;
		return 1;
	}
	return 0;
}

int main(int argc, char* argv[])
{
	caffe::CaffeBinding caffeBinding;
	kCaffeBinding = &caffeBinding;
		
	try 
	{
		std::string input_path;
		std::string output_xml_path;
		std::string dataset_name;
		std::string dataset_comment;
		int num_frames = 5;

		if (process_argument(argc, argv, input_path, output_xml_path, dataset_name, dataset_comment, num_frames) == 0)
		{

			cout << "Dataset name:" << dataset_name << endl;
			cout << "Dataset comment:" << dataset_comment << endl;

			cout << num_frames << endl;
			
			fs::path frames_path;
			if (!fs::is_directory(input_path))
			{
				cout << "Input video is: " << input_path << endl;
				cout << "Splitting input video to images..." << endl;
				frames_path = split_video_to_frames(input_path, num_frames);
			}
			else 
			{
				frames_path = input_path;
			}

			
			cout << "Running face detection..." << endl;
			run_face_detection(frames_path, output_xml_path, dataset_name, dataset_comment);

			cout << "Results are written to: " << output_xml_path << endl;
		}
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
		return 1;
	}

	return 0;
}

