# 26-1 YBIGTA DS Mini Team Project

YBIGTA 28기 DS 미니 프로젝트의 일환으로 진행한 논문 구현 프로젝트입니다.

본 프로젝트는 Vision-Language Model(VLM)의 핵심 구조를 직접 코드로 구현하며, 논문에서 제안된 주요 아이디어가 실제 코드 구조로 어떻게 연결되는지 이해하는 것을 목표로 합니다.

## Project Overview

본 레포지토리는 BLIP 논문의 핵심 구조를 학습용으로 단순화하여 구현한 프로젝트입니다.

원 논문:

**BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**

BLIP은 이미지와 텍스트를 함께 다루는 Vision-Language Pre-training(VLP) 프레임워크입니다.  
기존 모델들이 주로 이해 기반 태스크 또는 생성 기반 태스크 중 하나에 강점을 보였던 것과 달리, BLIP은 이미지-텍스트 이해와 텍스트 생성 태스크를 하나의 구조 안에서 함께 다룰 수 있도록 설계되었습니다.

이 프로젝트에서는 논문 전체를 완전 재현하기보다는, BLIP의 핵심 구조인 MED(Multimodal Mixture of Encoder-Decoder)를 작은 규모로 구현하는 데 초점을 두었습니다.

## Affiliation

- Organization: YBIGTA
- Track: Data Science
- Generation: 28기
- Contributor: 문형서

## Main Objective

이 프로젝트의 목적은 단순히 논문 결과를 재현하는 것이 아니라, 논문 속 모델 구조를 코드 레벨에서 이해하는 것입니다.

특히 다음 질문들에 답할 수 있도록 구현을 구성했습니다.

- 이미지는 어떻게 Transformer가 처리할 수 있는 token sequence로 바뀌는가?
- 텍스트는 어떻게 embedding과 Transformer block을 거쳐 표현되는가?
- 이미지와 텍스트는 어떤 방식으로 같은 의미 공간에 정렬되는가?
- image-text matching은 어떤 구조로 계산되는가?
- image-conditioned language modeling은 어떻게 구현되는가?
- BLIP의 ITC, ITM, LM objective는 코드상에서 어떻게 연결되는가?

## Implemented Components

현재 구현은 BLIP의 핵심 흐름을 반영한 미니 버전입니다.

```text
BLIP-mini
├─ Image Encoder
│  └─ Vision Transformer
│
├─ Text Encoder
│  └─ Transformer-based text representation
│
├─ Image-grounded Text Encoder
│  └─ Cross-attention for image-text matching
│
├─ Image-grounded Text Decoder
│  └─ Causal self-attention + cross-attention for language modeling
│
└─ Losses
   ├─ ITC: Image-Text Contrastive Loss
   ├─ ITM: Image-Text Matching Loss
   └─ LM: Language Modeling Loss