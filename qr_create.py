import qrcode
qr_img = qrcode.make('https://github.com/eglabsid/AutonomousQA_Tool')
qr_img.save('autuqa_qr.png')