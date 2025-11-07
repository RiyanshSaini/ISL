import tensorflow as tf
import os

def fix_and_convert_model():
    try:
        print("🔄 Loading model with compile=False to avoid optimizer issues...")
        
        # Load model without compilation
        model = tf.keras.models.load_model(
            r"D:\isl-web\model\model.h5", 
            compile=False
        )
        
        print("✅ Model loaded successfully!")
        print("📊 Model architecture:")
        model.summary()
        
        # Recompile with simple optimizer for compatibility
        print("🔄 Recompiling with compatible optimizer...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        # Convert all weights to float32
        print("🔄 Converting weights to float32...")
        for layer in model.layers:
            if layer.weights:
                weights = layer.get_weights()
                new_weights = [w.astype('float32') for w in weights]
                layer.set_weights(new_weights)
        
        # Save the fixed model
        output_path = "model_float32.h5"
        model.save(output_path)
        
        print(f"✅ Successfully saved fixed model as: {output_path}")
        
        # Verify the new model can be loaded
        print("🔄 Verifying the fixed model...")
        test_model = tf.keras.models.load_model(output_path, compile=False)
        print("✅ Fixed model verified and ready for conversion!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_and_convert_model()
    if success:
        print("\n🎉 Model fixed successfully! Now convert to TensorFlow.js:")
        print("tensorflowjs_converter --input_format=keras model_float32.h5 model/")
    else:
        print("\n💥 Model conversion failed.")