---
layout: post
title: airbnb 클론 프로젝트 - Amenity Detection
subtitle: Custom Dataset에 대한 Object Detection 프로젝트
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/amenity_thumb.png
share-img: /assets/img/path.jpg
tags: [object detection, custom dataset, centernet]
---

airbnb 클론 프로젝트로써 최신 Object Detection 딥러닝 모델인 CenterNet을 이용한 Custom Dataset에 대한 Object Detection 프로젝트를 진행하였습니다.

## Object Detection 문제영역 소개

Object Detection은 컴퓨터 비전 분야의 대표적인 문제영역으로 물체 내에 존재하는 물체들의 위치와 물체의 종류를 검출하는 문제 영역입니다. 

이때 Object Detection의 예측 결과로 출력되는 것은 한 이미지 내에 존재하는 물체들의 위치를 나타내는 **[Bounding Box의 꼭지점 좌표(x_min,y_min,x_max,y_max), 물체의 label, 검출 결과에 대한 확신의 정도]** 형태가 됩니다.

