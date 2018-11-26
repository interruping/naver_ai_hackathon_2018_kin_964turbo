# Naver AI Hackathon 2018 - 964 Turbo 

 964turbo팀의  [네이버 지식iN 질문 유사도 예측](https://github.com/naver/ai-hackathon-2018/blob/master/missions/kin.md)  미션 결승 모델입니다.

##  Model Summary
![model](https://github.com/interruping/naver_ai_hackathon_2018_kin_964turbo/blob/master/img/model.png?raw=true)
- keras를 사용하여 모델을 개발하였습니다.
- 별도의 Embedding Layer를 구성하지 않고, 문장을 단어로 쪼갠 뒤 일일이 Word2Vec로 전처리하였습니다.
- 단어 전처리에 사용한 Word2Vec 모델은 직접 학습시키지 않고, [학습된(pre-trained) 모델](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)을 사용하였습니다. 
-  모델은 Ma-LSTM을 사용하였으며, [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)  논문을 참고하였습니다.
- 맨하탄 거리를 구하는  Custom Layer 코드는 https://github.com/likejazz/Siamese-LSTM 에서 가져왔습니다.

##  Final Result

- Accuracy 0.77478 ( 13 위 )

![result](https://github.com/interruping/naver_ai_hackathon_2018_kin_964turbo/blob/master/img/result_kin_final.png?raw=true)

## License

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 사항은 [LICENSE](https://github.com/interruping/naver_ai_hackathon_2018_kin_964turbo/raw/master/LICENSE)파일을 참조하세요.
