package com.nkastanos.mnn;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("MNNJni");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());

        runMnistDemo();
    }

    /**
     * A native method that is implemented by the 'MNNJni' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public native int runMnistDemo();

}