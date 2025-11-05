"""
Model Selector Module for Anomalib

이 모듈은 Anomalib 모델을 선택하고 설정하는 유연한 방법을 제공합니다. 
모델의 요구사항에 따라 동적으로 설정됩니다.
- 검증 세트가 필요하지 않은 모델 (예: PatchCore)
- 다른 학습 루프를 가진 모델
- 동적 모델 설정
"""

from typing import Optional, Dict, Any, Tuple
from anomalib.models import get_model
import logging

# 로거 설정
try:
    logger = logging.getLogger("HybridDetector.model_selector")
except:
    import logging
    logger = logging.getLogger(__name__)


class ModelSelector:
    """
    모델 선택 및 동적 설정을 위한 클래스입니다.
    
    이 클래스는 다음을 제공합니다:
    1. 모델 이름(문자열 입력)에 따른 선택
    2. 모델 요구사항 확인 (검증 세트, 등)
    3. 다른 학습 루프를 위한 설정
    
    examples:
    - patchcore:
        {
            'use_validation': False,
            'use_mask': False,
            'monitor_metric': None,
            'check_val_every_n_epoch': None,
        }
    - padim:
        {
            'use_validation': True,
            'use_mask': False,
            'monitor_metric': 'image_AUROC',
            'check_val_every_n_epoch': 1,
        }
    - cfa:
        {
            'use_validation': True,
            'use_mask': False,
            'monitor_metric': 'image_AUROC',
            'check_val_every_n_epoch': 1,
        }
    """
    
    # 모델 요구사항 매핑
    # 검증 세트가 필요하지 않은 모델
    MODELS_WITHOUT_VALIDATION = {
        'patchcore',
        'padim',
        'efficient_ad',
        'fastflow',
    }
    
    # 검증 세트가 필요한 모델
    MODELS_WITH_VALIDATION = {
        'cfa',
        'cflow',
        'dfkde',
        'dfm',
        'draem',
        'ganomaly',
        'reverse_distillation',
        'stfpm',
        'uflow',
        'winclip',
    }
    
    # 마스크가 필요한 모델
    MODELS_REQUIRING_MASKS = {
        'draem',
        'ganomaly',
    }
    
    def __init__(self):
        """ModelSelector 초기화."""
        self._model_cache: Dict[str, Any] = {}
        logger.debug("ModelSelector 초기화 완료")
    
    def get_model(self, model_name: str, **model_kwargs) -> Any:
        """
        모델 이름에 따른 Anomalib 모델을 가져옵니다.
        
        Args:
            model_name: 모델 이름(string) - 예: 'patchcore', 'padim', 등
            **model_kwargs: get_model()에 전달할 추가 키워드 인수
        
        Returns:
            Anomalib model instance
        
        Raises:
            ValueError: model_name이 유효하지 않거나 모델을 로드할 수 없는 경우
        """
        if not isinstance(model_name, str):
            raise ValueError(f"model_name은 문자열이어야 합니다, {type(model_name)}을 받았습니다.")
        
        model_name_lower = model_name.lower()
        
        # 캐시 확인
        cache_key = f"{model_name_lower}_{hash(str(sorted(model_kwargs.items())))}"
        if cache_key in self._model_cache:
            logger.debug(f"캐시된 모델을 반환합니다: {model_name}")
            return self._model_cache[cache_key]
        
        try:
            # Anomalib에서 모델 가져오기
            model = get_model(model_name_lower, **model_kwargs)
            logger.info(f"모델을 성공적으로 로드했습니다: {model_name}")
            
            # 모델 캐시
            self._model_cache[cache_key] = model
            
            return model
        except Exception as e:
            logger.error(f"모델을 로드하는 데 실패했습니다: '{model_name}': {e}")
            raise ValueError(f"모델을 로드하는 데 실패했습니다: '{model_name}': {e}")
    
    def requires_validation(self, model_name: str, model_instance: Optional[Any] = None) -> bool:
        """
        모델이 검증 세트를 필요로 하는지 확인합니다.
        
        Args:
            model_name: 모델 이름(string)
            model_instance: 동적으로 확인할 선택적 모델 인스턴스
        
        Returns:
            검증 세트를 필요로 하는 경우 True, 그렇지 않으면 False 
        
        examples:
        - patchcore: 검증 세트를 필요로 하지 않음
        - padim: 검증 세트를 필요로 함
        - cfa: 검증 세트를 필요로 함
        - draem: 검증 세트를 필요로 함
        - ganomaly: 검증 세트를 필요로 함
        - reverse_distillation: 검증 세트를 필요로 함
        - stfpm: 검증 세트를 필요로 함
        - uflow: 검증 세트를 필요로 함
        - winclip: 검증 세트를 필요로 함
        - efficient_ad: 검증 세트를 필요로 함
        - fastflow: 검증 세트를 필요로 함
        """
        model_name_lower = model_name.lower()
        
        # 모델명 대조
        if model_name_lower in self.MODELS_WITHOUT_VALIDATION:
            return False
        if model_name_lower in self.MODELS_WITH_VALIDATION:
            return True
        
        # 모델 인스턴스가 제공된 경우, hasattr를 사용하여 동적으로 확인
        if model_instance is not None:
            # 모델이 특정 속성을 가지고 있는지 확인
            # 모델이 'requires_validation' 속성을 가지고 있는지 확인
            if hasattr(model_instance, 'requires_validation'):
                return bool(model_instance.requires_validation)
            
            # 모델이 검증 관련 메서드를 가지고 있는지 확인
            if hasattr(model_instance, 'validation_step'):
                return True
            
            # 모델이 'val_loss' 또는 유사한 속성을 가지고 있는지 확인
            if hasattr(model_instance, 'hparams'):
                hparams = model_instance.hparams
                if hasattr(hparams, 'val_split_ratio') and hparams.val_split_ratio > 0:
                    return True
        
        # 기본: 검증이 필요하다고 가정 (안전한 기본값)
        logger.warning(
            f"모델 '{model_name}'은 알려진 매핑에 없습니다. "
            "검증이 필요하다고 가정합니다. MODELS_WITHOUT_VALIDATION 또는 MODELS_WITH_VALIDATION에 추가하세요."
            "모델의 요구사항을 알고 있다면 MODELS_WITHOUT_VALIDATION 또는 MODELS_WITH_VALIDATION에 추가하세요."
        )
        return True
    
    def requires_mask(self, model_name: str, model_instance: Optional[Any] = None) -> bool:
        """
        모델이 마스크 데이터를 필요로 하는지 확인합니다.
        
        Args:
            model_name: 모델명 (string)
            model_instance: 동적으로 확인할 선택적 모델 인스턴스
        
        Returns:
            마스크 데이터를 필요로 하는 경우 True, 그렇지 않으면 False 
        
        examples:
        - draem: 마스크 데이터를 필요로 함
        - ganomaly: 마스크 데이터를 필요로 함
        - efficient_ad: 마스크 데이터를 필요로 함
        - fastflow: 마스크 데이터를 필요로 함
        - patchcore: 마스크 데이터를 필요로 하지 않음
        - padim: 마스크 데이터를 필요로 하지 않음
        - cfa: 마스크 데이터를 필요로 하지 않음
        - cflow: 마스크 데이터를 필요로 하지 않음
        - dfkde: 마스크 데이터를 필요로 하지 않음
        - dfm: 마스크 데이터를 필요로 하지 않음
        - stfpm: 마스크 데이터를 필요로 하지 않음
        - uflow: 마스크 데이터를 필요로 하지 않음
        - winclip: 마스크 데이터를 필요로 하지 않음
        """
        model_name_lower = model_name.lower()
        
        # 모델명 대조
        if model_name_lower in self.MODELS_REQUIRING_MASKS:
            return True
        
        # 모델 인스턴스가 제공된 경우, hasattr를 사용하여 동적으로 확인
        if model_instance is not None:
            # 모델이 마스크 관련 속성을 가지고 있는지 확인
            if hasattr(model_instance, 'requires_mask'):
                return bool(model_instance.requires_mask)
            
            # 모델이 마스크 요구사항을 가지고 있는지 확인
            if hasattr(model_instance, 'hparams'):
                hparams = model_instance.hparams
                if hasattr(hparams, 'use_mask') and hparams.use_mask:
                    return True
        
        return False
    
    def get_model_config(self, model_name: str, model_instance: Optional[Any] = None) -> Dict[str, Any]:
        """
        모델의 요구사항에 따른 설정 딕셔너리를 가져옵니다.
        
        Args:
            model_name: 모델 이름(string)
            model_instance: 동적으로 확인할 선택적 모델 인스턴스
        
        Returns:
            모델 설정 딕셔너리:
            {
                'requires_validation': bool,
                'requires_mask': bool,
                'training_mode': str,  # 'with_validation' 또는 'without_validation'
            }
            
        examples:
        - patchcore:
            {
                'requires_validation': False,
                'requires_mask': False,
                'training_mode': 'without_validation',
            }
        - padim:
            {
                'requires_validation': True,
                'requires_mask': False,
                'training_mode': 'with_validation',
            }
        - cfa:
            {
                'requires_validation': True,
                'requires_mask': False,
                'training_mode': 'with_validation',
        """
        
        requires_val = self.requires_validation(model_name, model_instance)
        requires_mask = self.requires_mask(model_name, model_instance)
        
        config = {
            'requires_validation': requires_val,
            'requires_mask': requires_mask,
            'training_mode': 'with_validation' if requires_val else 'without_validation',
        }
        
        logger.info(
            f"Model '{model_name}' configuration: "
            f"validation={requires_val}, mask={requires_mask}"
        )
        
        return config
    
    def get_training_config(self, model_name: str, model_instance: Optional[Any] = None) -> Dict[str, Any]:
        """
        모델의 학습 관련 설정 딕셔너리를 가져옵니다.
        
        Args:
            model_name: 모델 이름(string)
            model_instance: 동적으로 확인할 선택적 모델 인스턴스
        
        Returns:
            학습 설정 딕셔너리:
            {
                'use_validation': bool,
                'use_mask': bool,
                'monitor_metric': str,
                'check_val_every_n_epoch': int,
            }
        
        examples:
        - patchcore:
            {
                'use_validation': False,
                'use_mask': False,
                'monitor_metric': None,
                'check_val_every_n_epoch': None,
            }
        - padim:
            {
                'use_validation': True,
                'use_mask': False,
                'monitor_metric': 'image_AUROC',
                'check_val_every_n_epoch': 1,
            }
        - cfa:
            {
                'use_validation': True,
                'use_mask': False,
                'monitor_metric': 'image_AUROC',
                'check_val_every_n_epoch': 1,
            }
        """
        model_config = self.get_model_config(model_name, model_instance)
        
        training_config = {
            'use_validation': model_config['requires_validation'],
            'use_mask': model_config['requires_mask'],
            'monitor_metric': 'image_AUROC' if model_config['requires_validation'] else None,
            'check_val_every_n_epoch': 1 if model_config['requires_validation'] else None,
        }
        
        return training_config


# Convenience function for easy access
def get_model_selector() -> ModelSelector:
    """ModelSelector 인스턴스를 가져옵니다."""
    if not hasattr(get_model_selector, '_instance'):
        get_model_selector._instance = ModelSelector()
    return get_model_selector._instance


# example usage
if __name__ == "__main__":
    # selector 초기화
    selector = ModelSelector()
    
    # 다른 모델 테스트
    test_models = ['patchcore', 'padim', 'cfa', 'draem']
    
    for model_name in test_models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_name}")
        print(f"{'='*50}")
        
        try:
            # 모델 가져오기
            model = selector.get_model(model_name)
            
            # 모델 설정 가져오기
            config = selector.get_model_config(model_name, model)
            print(f"Configuration: {config}")
            
            # 학습 설정 가져오기
            training_config = selector.get_training_config(model_name, model)
            print(f"Training config: {training_config}")
            
        except Exception as e:
            print(f"Error: {e}")

