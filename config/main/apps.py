# from django.apps import AppConfig


# class MainConfig(AppConfig):
#     default_auto_field = "django.db.models.BigAutoField"
#     name = "main"

# main/apps.py
import os
from django.apps import AppConfig
from django.conf import settings
class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    # 서버 시작 시 실행
    # def ready(self):
    #     from .rag_model.rag_pipeline import global_qa_chain, initialize_qa_chain
    #     # from . import views # 순환 참조 방지를 위해 내부에서 import
    #     if global_qa_chain is None:
    #         global_qa_chain = initialize_qa_chain()
    #         print("✨ RAG QA Chain 초기화 완료 및 Django 서버 로드됨.")
    def ready(self):
        # StatReloader의 중복 로드 방지 조건문 추가
        if os.environ.get('RUN_MAIN', None) != 'true':
             return
             
        from .rag_model import rag_pipeline

        if not settings.RAG_CHAIN_STORE.get('qa_chain'):
            print(">>> RAG Chain 초기화 시작 및 settings에 저장합니다...")
            qa_chain = rag_pipeline.initialize_qa_chain()
            
            if qa_chain:
                # ✅ 성공 시 settings에 저장합니다.
                settings.RAG_CHAIN_STORE['qa_chain'] = qa_chain
                print("✨ RAG QA Chain 최종 성공적으로 로드 완료.")
            else:
                # 실패 시 None 또는 에러 메시지 저장 (디버깅 목적)
                settings.RAG_CHAIN_STORE['qa_chain'] = None
                print("💥 RAG QA Chain 로드 실패! 챗봇 기능이 비활성화됩니다.")
