import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import javax.imageio.ImageIO;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.VideoInputFrameGrabber;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class main {
	public static void main(String[] args) throws Exception {
		CanvasFrame canvas = new CanvasFrame("WebCam Demo");
		canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
		VideoInputFrameGrabber webcam = new VideoInputFrameGrabber(0); // 1 for
																		// next
																		// camera
		FrameGrabber grabber = webcam;
		grabber.start();
		OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
		Frame frame;
		File file = new File("aaas.xml");
		String classifierName = file.getAbsolutePath();
		Loader.load(opencv_objdetect.class);
		Pointer p = cvLoad(classifierName);
		CvHaarClassifierCascade classifier = new CvHaarClassifierCascade(p);
		// 假如載入HaarClassifierCascade 失敗則結束程式
		if (classifier.isNull()) {
			System.err.println("Error loading classifier file \""
					+ classifierName + "\".");
			System.exit(1);
		}
		IplImage img;
		CvMemStorage storage = CvMemStorage.create();
		IplImage grayImage;
		canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());
		while (true) {
			frame = grabber.grab();
			img = converter.convert(frame);
			// 取得影像的長寬
			int width = img.width();
			int height = img.height();
			// 建立一個灰階影像（用IPL_DEPTH_8U 原因為，灰階影像每一個pixel R,G,B值皆相等，只需要存一份即可）
			grayImage = IplImage.create(width, height, IPL_DEPTH_8U, 1);
			cvClearMemStorage(storage);
			cvCvtColor(img, grayImage, CV_BGR2GRAY);
			org.bytedeco.javacpp.opencv_core.CvSeq faces = cvHaarDetectObjects(
					grayImage, classifier, storage, 1.1, 3,
					CV_HAAR_DO_CANNY_PRUNING);
			int total = faces.total();
            //在人臉上畫一個紅色的矩型
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                cvRectangle(img, cvPoint(x, y), cvPoint(x + w, y + h), CvScalar.RED, 1, CV_AA, 0);
            }
			canvas.showImage(converter.convert(img));
		}
	}

	public static BufferedImage IplImageToBufferedImage(IplImage src) {
		OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
		Java2DFrameConverter paintConverter = new Java2DFrameConverter();
		Frame frame = grabberConverter.convert(src);
		return paintConverter.getBufferedImage(frame, 1);
	}

}