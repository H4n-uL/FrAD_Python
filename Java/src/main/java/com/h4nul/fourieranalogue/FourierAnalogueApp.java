package com.h4nul.fourieranalogue;

import org.python.core.PyFunction;
import org.python.core.PyInteger;
import org.python.core.PyObject;
import org.python.core.PyString;
import org.python.util.PythonInterpreter;

public class FourierAnalogueApp {
    private static PythonInterpreter python;
    public static void main(String[] args) throws Exception {
        System.setProperty("python.import.site", "false");
        python = new PythonInterpreter();
        String encoderFile = FourierAnalogueApp.class.getResource("/jython/encoder.py").getPath();
        python.execfile(encoderFile);
        System.out.println("Python running");
        python.exec("print('python running')");
        PyFunction pyFuntion = (PyFunction) python.get("enc", PyFunction.class);

        PyObject pyobj = pyFuntion.__call__(new PyString("/Users/H4nUL/Desktop/target_for_love.mp3"), new PyInteger(64));
        System.out.println(pyobj.toString());

        // return pyobj.toString();
    }
}
