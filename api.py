from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import json
from datetime import datetime
from optimizer import RidePricingOptimizer

def create_pricing_api():
    """
    Creates a Flask web API for the ride pricing optimizer
    """
    app = Flask(__name__)
    CORS(app)  
    
    # Initialize the optimizer
    optimizer = RidePricingOptimizer()
    
    @app.route('/api/optimize', methods=['POST'])
    def optimize_pricing():
        """
        Main optimization endpoint
        
        Request body:
        {
            "demands": [20, 50, 80, 60, 40],
            "algorithm": "forward_greedy",  // optional, defaults to "forward_greedy"
            "cost_per_ride": 3              // optional, defaults to 3
        }
        
        Response:
        {
            "success": true,
            "data": {
                "optimal_prices": [9, 12, 15, 12, 9],
                "total_profit": 450.50,
                "algorithm_used": "forward_greedy",
                "execution_time_ms": 12.5,
                "detailed_analysis": [...],
                "timestamp": "2024-03-15T10:30:00Z"
            }
        }
        """
        try:
            # parse data into json
            data = request.get_json()
            
            # check if data is valid
            if not data or 'demands' not in data:
                return jsonify({
                    "success": False,
                    "error": "no demand data given"
                }), 400
            
            demands = data['demands']
            algorithm = data.get('algorithm', 'forward_greedy')
            cost_per_ride = data.get('cost_per_ride', 3)
            
            # check if demands is a list
            if not isinstance(demands, list):
                return jsonify({
                    "success": False,
                    "error": "Demands must be a non-empty list"
                }), 400

            
            # running alg
            optimizer.cost_per_ride = cost_per_ride

            start_time = time.time()
            optimal_prices = optimizer.optimize_prices(demands, algorithm=algorithm)
            execution_time = (time.time() - start_time) * 1000  # convert to ms
        
            total_profit = optimizer.simulate_day(demands, optimal_prices)
            
            return jsonify({
                "success": True,
                "data": {
                    "optimal_prices": optimal_prices,
                    "total_profit": total_profit,
                    "algorithm_used": algorithm,
                    "execution_time_ms": execution_time,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            })
        
        # return any errors that arise
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "error": {str(e)}
            }), 500
    

    @app.route('/api/check', methods=['GET'])
    def check():
        """
        checking if server is responsive
        """
        return jsonify({
            "success": True,
            "status": "active",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
    
    
    return app


def run_api_server(host='0.0.0.0', port=5000, debug=True):
    """
    run api server at port 5000
    """
    app = create_pricing_api()
    print("="*50)
    print(f"Starting server API")
    print(f"server: http://{host}:{port}")
    print("="*50)
    app.run(host=host, port=port, debug=debug)



run_api_server()

'''

 # curl command to test api 

curl -X POST http://127.0.0.1:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "demands": [20, 45, 85, 90, 70, 35, 25],
    "algorithm": "forward_greedy",
    "cost_per_ride": 3
  }

'''