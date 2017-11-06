/**
 * @fileoverview Training and serving test for a basic policy gradient
 *               reinforcement learning model to power the Plane Strike game.
 *               Using convnetjs since there is JS binding for Tensorflow.
 *
 *               Code is heavily based on: http://efavdb.com/battleship/.
 *               Read that page first if trying to follow.
 *
 *               Trained model 'neuralnet.json' is used for serving.
 *
 *               Avg game length is around 10.8.
 * @author: Wei Wei (weiwe@google.com)
 */

var convnetjs = require('convnetjs');
var jsonfile = require('jsonfile');
var math = require('mathjs');

const BOARD_WIDTH = 6;
const BOARD_HEIGHT = 6;
const BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH;
const PLANE_SIZE = 8;
const TRAINING = true;
const ITERATIONS = 20000;
const PRINT_INTERVAL = 100;
const WINDOW_SIZE = 50;
const NN_WEIGHT_FILE = 'neuralnet.json';

const ALPHA = 0.005;
const GAMMA = 0.5;

/**
* Generate a random integer within a specified range
*
* @param {number} min Lower bound of the range
* @param {number} max Upper bound of the range
* @return {number} return a random integer
*/
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}

/**
* Sample a number based on probabilities
*
* @param {array} probs probabilities
* @return {number} sampled number
*/
function randomSample(probs) {
  var randNum = Math.random();
  var s = 0;

  for (var i = 0; i < probs.length - 1; ++i) {
    s += probs[i];
    if (randNum < s) {
      return i;
    }
  }

  return (probs.length - 1);
}

/**
* Initialize the game
*
* @return {object} initialized game state
*/
function initPlane() {
  var hidden_board = new Array(BOARD_HEIGHT);
  var game_board = new Array(BOARD_HEIGHT);
  for (var i = 0; i < BOARD_HEIGHT; i++) {
    hidden_board[i] = new Array(BOARD_WIDTH);
    game_board[i] = new Array(BOARD_WIDTH);
    for (var j = 0; j < BOARD_WIDTH; j++) {
      hidden_board[i][j] = 0;
      game_board[i][j] = 0;
    }
  }

  // Populate the plane's position
  // First figure out the plane's orientation
  //   0: heading right
  //   1: heading up
  //   2: heading left
  //   3: heading down
  plane_orientation = getRandomInt(0, 4);

  // Figrue out plane core's position as the '*' below
  //        |         | |
  //       -*-        \-*-
  //        |         | |
  //       ---
  switch(plane_orientation) {
    case 0:
      plane_core_row = getRandomInt(1, BOARD_HEIGHT-1);
      plane_core_column = getRandomInt(2, BOARD_WIDTH-1);
      // Populate the tail
      hidden_board[plane_core_row][plane_core_column-2] = 1;
      hidden_board[plane_core_row-1][plane_core_column-2] = 1;
      hidden_board[plane_core_row+1][plane_core_column-2] = 1;
      break;
    case 1:
      plane_core_row = getRandomInt(1, BOARD_HEIGHT-2);
      plane_core_column = getRandomInt(1, BOARD_WIDTH-1);
      // Populate the tail
      hidden_board[plane_core_row+2][plane_core_column] = 1;
      hidden_board[plane_core_row+2][plane_core_column+1] = 1;
      hidden_board[plane_core_row+2][plane_core_column-1] = 1;
      break;
    case 2:
      plane_core_row = getRandomInt(1, BOARD_HEIGHT-1);
      plane_core_column = getRandomInt(1, BOARD_WIDTH-2);
      // Populate the tail
      hidden_board[plane_core_row][plane_core_column+2] = 1;
      hidden_board[plane_core_row-1][plane_core_column+2] = 1;
      hidden_board[plane_core_row+1][plane_core_column+2] = 1;
      break;
    case 3:
      plane_core_row = getRandomInt(2, BOARD_HEIGHT-1);
      plane_core_column = getRandomInt(1, BOARD_WIDTH-1);
      // Populate the tail
      hidden_board[plane_core_row-2][plane_core_column] = 1;
      hidden_board[plane_core_row-2][plane_core_column+1] = 1;
      hidden_board[plane_core_row-2][plane_core_column-1] = 1;
      break;
  }
  // Populate the cross
  hidden_board[plane_core_row][plane_core_column] = 1;
  hidden_board[plane_core_row+1][plane_core_column] = 1;
  hidden_board[plane_core_row-1][plane_core_column] = 1;
  hidden_board[plane_core_row][plane_core_column+1] = 1;
  hidden_board[plane_core_row][plane_core_column-1] = 1;

  return {
    'hidden_board' : hidden_board,
    'game_board'   : game_board
  };
}

/**
* Play a complete game and gather the logs
*
* @param {bool} training whether it's training or serving
* @return {object} game logs
*/
function playGame(training) {
  var init_state = initPlane();
  var hidden_board = init_state.hidden_board;
  var game_board = init_state.game_board;
  if (training == false) {
    console.log(hidden_board);
  }
  var board_pos_log = [];
  var action_log = [];
  var hit_log = [];
  var hits = 0;

  while (hits < PLANE_SIZE && action_log.length < BOARD_SIZE) {
    var flattened_game_board = [];
    for (var i = 0; i < BOARD_HEIGHT; i++) {
      for (var j = 0; j < BOARD_WIDTH; j++) {
        flattened_game_board.push(game_board[i][j]);
      }
    }
    board_pos_log.push(flattened_game_board);
    var flattened_game_board_vol = new convnetjs.Vol(flattened_game_board);
    var probs = trainer.net.forward(flattened_game_board_vol).w;
    var total_prob = 0;
    for (var i = 0; i < BOARD_SIZE; i++) {
      if (action_log.indexOf(i) != -1)
        probs[i] = 0;
      else
        total_prob += probs[i];
    }
    for (var i = 0; i < BOARD_SIZE; i++) {
      probs[i] /= total_prob;
    }
    var strike_pos = 0;
    if (training) {
      var index_array = [];
      var prob_array = [];
      for (var i = 0; i < BOARD_SIZE; i++) {
        index_array.push(i);
        prob_array.push(probs[i]);
      }
      strike_pos = math.pickRandom(index_array, prob_array);
    }
    else {
      // argmax
      var max = probs[0];
      var idx = 0;
      for (var i = 1; i < BOARD_SIZE; i++) {
        if (probs[i] > max) {
          max = probs[i];
          idx = i;
        }
      }
      strike_pos = idx;
    }
    var x = Math.floor(strike_pos / BOARD_WIDTH);
    var y = strike_pos % BOARD_WIDTH;

    if (hidden_board[x][y] == 1) {
      hits += 1;
      game_board[x][y] = 1;
      hit_log.push(1);
    }
    else {
      game_board[x][y] = -1;
      hit_log.push(0);
    }
    action_log.push(strike_pos);
    if (training == false)
      console.log(x + ', ' + y + ' *** ' + hit_log[hit_log.length-1]);
  }
  return {
    'board_pos_log' : board_pos_log,
    'action_log'    : action_log,
    'hit_log'       : hit_log
  };
}

/**
* Calculate the rewards based on hit log
*
* @param {array} hit_log hit log from a complete game
* @param {number} gamma hyperparameter
* @return {object} reward of each action
*/
function calculateRewards(hit_log, gamma) {
  var s;
  var hit_log_weighted = [];
  var rewards = [];
  for (var i = 0; i < hit_log.length; i++) {
    s = 0;
    for (var j = 0; j < i; j++) {
      s += hit_log[j];
    }
    hit_log_weighted.push((hit_log[i] - (PLANE_SIZE - s)/(BOARD_SIZE - i))
                            * Math.pow(gamma, i));
  }
  for (var i = 0; i < hit_log.length; i++) {
    s = 0;
    for (var j = i; j < hit_log_weighted.length; j++) {
      s += hit_log_weighted[j];
    }
    rewards.push(Math.pow(gamma, -i) * s);
  }
  return {'rewards' : rewards};
}

var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:36});
layer_defs.push({type:'fc', num_neurons:50, activation:'relu'});
layer_defs.push({type:'fc', num_neurons:100, activation:'relu'});
layer_defs.push({type:'softmax', num_classes:BOARD_SIZE});
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
var opt = {method:'sgd', momentum:0.0, batch_size:1, l1_decay:0, l2_decay:0};
var trainer = new convnetjs.SGDTrainer(net, opt);

var game_length = [];
var game_result = [];
var board_pos_log = [];
var action_log = [];
var hit_log = [];
var rewards_log= [];

if (TRAINING == true) {
  for (var i = 0; i < ITERATIONS; i++) {
    if (i % PRINT_INTERVAL == 0) {
      console.log(i);
      var json = net.toJSON();
      jsonfile.writeFileSync(NN_WEIGHT_FILE, JSON.stringify(json));
    }
    game_result = playGame(TRAINING);
    board_pos_log = game_result.board_pos_log;
    action_log = game_result.action_log;
    hit_log = game_result.hit_log;
    game_length.push(action_log.length);
    var result = calculateRewards(hit_log, GAMMA);
    rewards_log = result.rewards;

    for (var j = 0; j < action_log.length; j++) {
      trainer.learning_rate = ALPHA * rewards_log[j];
      var board_pos_vol = new convnetjs.Vol(board_pos_log[j]);
      trainer.train(board_pos_vol, action_log[j]);
    }
  }

  console.log('*******');
  // lacking a good way to visualize the progress in js
  var running_avg_length = [];
  for (var i = 0; i < game_length.length; i++) {
    var s = 0;
    if (i < WINDOW_SIZE) {
      running_avg_length.push(game_length[i]);
      continue;
    }
    else
    {
      for (var j = 0; j < WINDOW_SIZE; j++) {
        s += game_length[i-j];
      }
    }
    running_avg_length.push(s/WINDOW_SIZE);
    console.log(s/WINDOW_SIZE);
  }
}
else {
  var json = JSON.parse(jsonfile.readFileSync(NN_WEIGHT_FILE));
  net.fromJSON(json);
  playGame(false);
}
