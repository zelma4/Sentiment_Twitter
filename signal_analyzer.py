#!/usr/bin/env python3
"""
Signal Analysis and Conflict Resolution
Analyzes conflicting trading signals and provides consensus recommendations
"""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Individual trading signal"""
    source: str
    direction: str  # UP, DOWN, HOLD
    confidence: float
    strength: float
    timestamp: datetime
    details: Dict

class SignalAnalyzer:
    """
    Advanced signal analysis with conflict resolution
    """
    
    def __init__(self):
        self.signal_weights = {
            'lightgbm': 0.35,      # High weight for ML prediction
            'technical': 0.25,      # Technical analysis
            'sentiment': 0.20,      # Sentiment analysis  
            'enhanced': 0.15,       # Enhanced metrics
            'advanced_neural': 0.40  # Highest weight when available
        }
        
    def analyze_signals(self, analysis_report: Dict) -> Dict:
        """
        Analyze all signals and resolve conflicts
        
        Args:
            analysis_report: Complete analysis report from bot
            
        Returns:
            Consensus analysis with recommendations
        """
        try:
            # Extract individual signals
            signals = self._extract_signals(analysis_report)
            
            if not signals:
                return {'error': 'No signals available for analysis'}
            
            # Calculate consensus
            consensus = self._calculate_consensus(signals)
            
            # Detect conflicts
            conflicts = self._detect_conflicts(signals)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(consensus, conflicts, signals)
            
            return {
                'consensus': consensus,
                'conflicts': conflicts,
                'signals': [self._signal_to_dict(s) for s in signals],
                'recommendations': recommendations,
                'analysis_timestamp': datetime.utcnow(),
                'signal_count': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Signal analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_signals(self, analysis_report: Dict) -> List[TradingSignal]:
        """Extract individual signals from analysis report"""
        signals = []
        
        try:
            # LightGBM Signal
            if analysis_report.get('lightgbm_prediction'):
                lgb = analysis_report['lightgbm_prediction']
                direction = lgb.get('direction_text', 'HOLD')
                confidence = lgb.get('confidence', 0.5)
                
                signals.append(TradingSignal(
                    source='lightgbm',
                    direction=direction,
                    confidence=confidence,
                    strength=confidence,
                    timestamp=datetime.utcnow(),
                    details=lgb
                ))
            
            # Technical Analysis Signal
            if analysis_report.get('technical'):
                tech = analysis_report['technical']
                recommendation = tech.get('recommendation', 'HOLD')
                strength = abs(tech.get('signals', {}).get('strength', 0))
                
                # Convert technical recommendation to direction
                if recommendation in ['STRONG_BUY', 'BUY']:
                    direction = 'UP'
                elif recommendation in ['STRONG_SELL', 'SELL']:
                    direction = 'DOWN'
                else:
                    direction = 'HOLD'
                
                signals.append(TradingSignal(
                    source='technical',
                    direction=direction,
                    confidence=min(strength, 1.0),
                    strength=strength,
                    timestamp=datetime.utcnow(),
                    details=tech
                ))
            
            # Sentiment Signal
            if analysis_report.get('sentiment'):
                sent = analysis_report['sentiment']
                overall = sent.get('overall', {})
                sentiment_score = overall.get('overall_score', 0)
                confidence = overall.get('confidence', 0.5)
                
                # Convert sentiment to direction
                if sentiment_score > 0.3:
                    direction = 'UP'
                elif sentiment_score < -0.3:
                    direction = 'DOWN'
                else:
                    direction = 'HOLD'
                
                signals.append(TradingSignal(
                    source='sentiment',
                    direction=direction,
                    confidence=confidence,
                    strength=abs(sentiment_score),
                    timestamp=datetime.utcnow(),
                    details=overall
                ))
            
            # Advanced Neural Signal (if available)
            if analysis_report.get('advanced_prediction'):
                adv = analysis_report['advanced_prediction']
                direction = adv.get('direction_text', 'HOLD')
                confidence = adv.get('confidence', 0.5)
                
                signals.append(TradingSignal(
                    source='advanced_neural',
                    direction=direction,
                    confidence=confidence,
                    strength=confidence,
                    timestamp=datetime.utcnow(),
                    details=adv
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error extracting signals: {e}")
            return []
    
    def _calculate_consensus(self, signals: List[TradingSignal]) -> Dict:
        """Calculate weighted consensus from all signals"""
        try:
            if not signals:
                return {'direction': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}
            
            # Calculate weighted scores
            weighted_scores = {'UP': 0.0, 'DOWN': 0.0, 'HOLD': 0.0}
            total_weight = 0.0
            
            for signal in signals:
                weight = self.signal_weights.get(signal.source, 0.1)
                weighted_score = weight * signal.confidence
                
                weighted_scores[signal.direction] += weighted_score
                total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                for direction in weighted_scores:
                    weighted_scores[direction] /= total_weight
            
            # Find strongest direction
            consensus_direction = max(weighted_scores, key=weighted_scores.get)
            consensus_confidence = weighted_scores[consensus_direction]
            
            # Calculate agreement (how many signals agree with consensus)
            agreeing_signals = [s for s in signals if s.direction == consensus_direction]
            agreement = len(agreeing_signals) / len(signals)
            
            return {
                'direction': consensus_direction,
                'confidence': consensus_confidence,
                'agreement': agreement,
                'scores': weighted_scores,
                'agreeing_signals': len(agreeing_signals),
                'total_signals': len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return {'direction': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}
    
    def _detect_conflicts(self, signals: List[TradingSignal]) -> Dict:
        """Detect conflicts between signals"""
        try:
            if len(signals) < 2:
                return {'has_conflicts': False, 'conflict_level': 0.0}
            
            directions = [s.direction for s in signals]
            unique_directions = set(directions)
            
            # No conflict if all signals agree
            if len(unique_directions) == 1:
                return {
                    'has_conflicts': False,
                    'conflict_level': 0.0,
                    'conflicting_pairs': []
                }
            
            # Analyze specific conflicts
            conflicting_pairs = []
            conflict_scores = []
            
            for i, signal1 in enumerate(signals):
                for j, signal2 in enumerate(signals[i+1:], i+1):
                    if signal1.direction != signal2.direction:
                        # Calculate conflict severity
                        conflict_severity = (signal1.confidence + signal2.confidence) / 2
                        conflicting_pairs.append({
                            'signal1': signal1.source,
                            'direction1': signal1.direction,
                            'signal2': signal2.source, 
                            'direction2': signal2.direction,
                            'severity': conflict_severity
                        })
                        conflict_scores.append(conflict_severity)
            
            avg_conflict_level = statistics.mean(conflict_scores) if conflict_scores else 0.0
            
            return {
                'has_conflicts': len(conflicting_pairs) > 0,
                'conflict_level': avg_conflict_level,
                'conflicting_pairs': conflicting_pairs,
                'unique_directions': len(unique_directions),
                'direction_distribution': {dir: directions.count(dir) for dir in unique_directions}
            }
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return {'has_conflicts': False, 'conflict_level': 0.0}
    
    def _generate_recommendations(self, consensus: Dict, conflicts: Dict, signals: List[TradingSignal]) -> Dict:
        """Generate trading recommendations based on analysis"""
        try:
            recommendations = {
                'primary_action': 'HOLD',
                'confidence_level': 'LOW',
                'risk_level': 'HIGH',
                'reasoning': [],
                'conditions': []
            }
            
            consensus_direction = consensus.get('direction', 'HOLD')
            consensus_confidence = consensus.get('confidence', 0.0)
            agreement = consensus.get('agreement', 0.0)
            has_conflicts = conflicts.get('has_conflicts', False)
            conflict_level = conflicts.get('conflict_level', 0.0)
            
            # Determine primary action
            if consensus_confidence >= 0.7 and agreement >= 0.7:
                recommendations['primary_action'] = consensus_direction
                recommendations['confidence_level'] = 'HIGH'
                recommendations['reasoning'].append(f"Strong consensus: {consensus_confidence:.1%} confidence, {agreement:.1%} agreement")
            elif consensus_confidence >= 0.6 and agreement >= 0.6:
                recommendations['primary_action'] = consensus_direction
                recommendations['confidence_level'] = 'MEDIUM'
                recommendations['reasoning'].append(f"Moderate consensus: {consensus_confidence:.1%} confidence, {agreement:.1%} agreement")
            else:
                recommendations['primary_action'] = 'HOLD'
                recommendations['confidence_level'] = 'LOW'
                recommendations['reasoning'].append(f"Weak consensus: {consensus_confidence:.1%} confidence, {agreement:.1%} agreement")
            
            # Adjust for conflicts
            if has_conflicts:
                if conflict_level >= 0.6:
                    recommendations['risk_level'] = 'VERY_HIGH'
                    recommendations['reasoning'].append(f"High conflict level: {conflict_level:.1%}")
                    if recommendations['primary_action'] != 'HOLD':
                        recommendations['conditions'].append("Consider smaller position size due to conflicting signals")
                elif conflict_level >= 0.4:
                    recommendations['risk_level'] = 'HIGH'
                    recommendations['reasoning'].append(f"Moderate conflict level: {conflict_level:.1%}")
                else:
                    recommendations['risk_level'] = 'MEDIUM'
                    recommendations['reasoning'].append(f"Low conflict level: {conflict_level:.1%}")
            else:
                if consensus_confidence >= 0.7:
                    recommendations['risk_level'] = 'LOW'
                else:
                    recommendations['risk_level'] = 'MEDIUM'
                recommendations['reasoning'].append("No conflicting signals detected")
            
            # Add specific signal insights
            high_confidence_signals = [s for s in signals if s.confidence >= 0.7]
            if high_confidence_signals:
                recommendations['reasoning'].append(f"High confidence signals: {[s.source for s in high_confidence_signals]}")
            
            # Add conditions based on signal quality
            if len(signals) < 3:
                recommendations['conditions'].append("Limited signal sources - consider waiting for more data")
            
            if any(s.source == 'advanced_neural' for s in signals):
                recommendations['conditions'].append("Advanced neural network prediction available")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'primary_action': 'HOLD', 'confidence_level': 'LOW', 'risk_level': 'HIGH'}
    
    def _signal_to_dict(self, signal: TradingSignal) -> Dict:
        """Convert signal to dictionary for JSON serialization"""
        return {
            'source': signal.source,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'strength': signal.strength,
            'timestamp': signal.timestamp.isoformat()
        }

def analyze_current_signals(analysis_report: Dict) -> Dict:
    """
    Main function to analyze current trading signals
    
    Args:
        analysis_report: Analysis report from main bot
        
    Returns:
        Complete signal analysis with recommendations
    """
    analyzer = SignalAnalyzer()
    return analyzer.analyze_signals(analysis_report)

def print_signal_analysis(analysis: Dict):
    """Pretty print signal analysis results"""
    if 'error' in analysis:
        print(f"❌ Analysis Error: {analysis['error']}")
        return
    
    print("🔍 TRADING SIGNAL ANALYSIS")
    print("=" * 50)
    
    # Consensus
    consensus = analysis.get('consensus', {})
    direction = consensus.get('direction', 'UNKNOWN')
    confidence = consensus.get('confidence', 0)
    agreement = consensus.get('agreement', 0)
    
    direction_emoji = "⬆️" if direction == 'UP' else "⬇️" if direction == 'DOWN' else "➡️"
    print(f"📊 CONSENSUS: {direction_emoji} {direction}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"   Agreement: {agreement:.1%}")
    
    # Conflicts
    conflicts = analysis.get('conflicts', {})
    if conflicts.get('has_conflicts'):
        conflict_level = conflicts.get('conflict_level', 0)
        print(f"⚠️  CONFLICTS DETECTED: {conflict_level:.1%} severity")
        
        for pair in conflicts.get('conflicting_pairs', [])[:3]:  # Show top 3
            print(f"   {pair['signal1']} ({pair['direction1']}) vs {pair['signal2']} ({pair['direction2']})")
    else:
        print("✅ NO CONFLICTS: All signals aligned")
    
    # Individual Signals
    print(f"\n📡 INDIVIDUAL SIGNALS ({analysis.get('signal_count', 0)}):")
    for signal in analysis.get('signals', []):
        source = signal['source']
        direction = signal['direction']
        confidence = signal['confidence']
        emoji = "⬆️" if direction == 'UP' else "⬇️" if direction == 'DOWN' else "➡️"
        print(f"   {source:12} {emoji} {direction:4} ({confidence:.1%})")
    
    # Recommendations
    recommendations = analysis.get('recommendations', {})
    action = recommendations.get('primary_action', 'HOLD')
    conf_level = recommendations.get('confidence_level', 'LOW')
    risk_level = recommendations.get('risk_level', 'HIGH')
    
    action_emoji = "🟢" if action == 'UP' else "🔴" if action == 'DOWN' else "🟡"
    print(f"\n🎯 RECOMMENDATION: {action_emoji} {action}")
    print(f"   Confidence: {conf_level}")
    print(f"   Risk Level: {risk_level}")
    
    reasoning = recommendations.get('reasoning', [])
    if reasoning:
        print(f"\n💡 REASONING:")
        for reason in reasoning:
            print(f"   • {reason}")
    
    conditions = recommendations.get('conditions', [])
    if conditions:
        print(f"\n⚠️  CONDITIONS:")
        for condition in conditions:
            print(f"   • {condition}")

# Example usage for testing
if __name__ == "__main__":
    # Simulate current server analysis
    test_analysis = {
        'lightgbm_prediction': {
            'direction_text': 'UP',
            'confidence': 0.667
        },
        'technical': {
            'recommendation': 'SELL',
            'signals': {'strength': -1.00}
        },
        'sentiment': {
            'overall': {
                'overall_score': 0.107,
                'confidence': 0.70
            }
        }
    }
    
    print("🧪 Testing with current server signals...")
    analysis = analyze_current_signals(test_analysis)
    print_signal_analysis(analysis)
