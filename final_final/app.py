from agent import agent

if __name__ == "__main__":
    img_path = r"C:\\Users\pc_37\\Desktop\\Final_Project\\CoSEV A cotton disease dataset for detection and classification of severity stages and multiple disease occurrence\\CoSev\\CoSev\\curl_stage2\\CS2 (1).jpeg"
    result = agent.run(f"Analyze this cotton leaf image: {img_path}")
    print(result)
