from edge_impulse_linux.runner import ImpulseRunner

print("Starting script...")

#model_path = "model/mission_model.eim"
model_path = "/Users/pranny/Desktop/IR_Product/gw_freq_edgeimpulse/model/mission_model.eim"   # model filename

runner = ImpulseRunner(model_path)

model_info = runner.init()
print("Model loaded successfully!")
print("Model info:", model_info)
