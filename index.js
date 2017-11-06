/**
 * @fileoverview Plane Strike game for Google Home,
 *               powered by reinforcement learning
 * @author: Wei Wei (weiwe@google.com)
 */
'use strict';

var express = require('express');
var bodyParser = require('body-parser');
var convnetjs = require('convnetjs');
var jsonfile = require('jsonfile');
// //dashbot analytics
//var //dashbot = require('//dashbot')
//                     ('iBSeHHArqeH5pa1M3bV9i2a6Juluz9zfS2S1YYT4').google;
// voicelabs analytics
//var VoiceInsights = require('voicelabs-assistant-sdk');
//const VI_APP_TOKEN = '888a3ab0-ca9f-11a6-0d08-02ddc10e4a8b';

// Constants
const BOARD_WIDTH = 6;
const BOARD_HEIGHT = 6;
const BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH;
const PLANE_SIZE = 8;
const NN_WEIGHT_FILE = 'neuralnet.json';
// Optimize user experience based on Tristan's input:
// Shorten prompts if users are exposed to the instructions twice
const VERBOSE_LIMIT = 1;


// Instantiate an Express application
var app = express();

app.use(bodyParser.json());

// Handle POST requests and call the planestrike() function
app.post('/planestrike', function (request, response) {
  planestrike(request, response);
  //dashbot.logIncoming(request.body);
});

// Start the server
var server = app.listen(process.env.PORT || '8080', function () {
  console.log('App listening on port %s', server.address().port);
  console.log('Press Ctrl+C to quit.');

  //VoiceInsights.initialize(VI_APP_TOKEN);
});

/**
 * Generate a random integer within a specified range
 *
 * @param {number} min Lower bound of the range
 * @param {number} max Upper bound of the range
 * @return {number} Return a random integer
 */
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}

/**
 * Initialize the game
 *
 * @return {object} Initialized game state
 */
function initPlaneStrike() {
  // Initiate 2 boards to track game state:
  //
  // Cell states in agent board:
  //   2: The cell is covered by the plane and the user has successfully
  //      hit it.
  //   1: The cell is covered by the plane but the user hasn't tried to
  //      strike at the cell
  //   0: Unexplored -- the user hasn't tried to strike at the cell
  //  -1: The user strikes at the cell but it's a miss
  //
  // Cell states in user board:
  //   1: The agent strikes at the cell and the user confirms that it's
  //      a hit
  //   0: Unexplored -- the agent hasn't tried to strike at the cell
  //  -1: The agent strikes at the cell but it's a miss

  var agent_board = new Array(BOARD_HEIGHT);
  var user_board = new Array(BOARD_HEIGHT);
  for (var i = 0; i < BOARD_HEIGHT; i++) {
    agent_board[i] = new Array(BOARD_WIDTH);
    user_board[i] = new Array(BOARD_WIDTH);
    for (var j = 0; j < BOARD_WIDTH; j++) {
      agent_board[i][j] = 0;
      user_board[i][j] = 0;
    }
  }

  // Populate the plane's position
  // First figure out the plane's orientation
  //   0: heading right
  //   1: heading up
  //   2: heading left
  //   3: heading down
  var plane_orientation = getRandomInt(0, 4);
  var plane_core_row = -1;
  var plane_core_column = -1;
  // Figure out plane core's position as the '*' below
  //        |         | |
  //       -*-        \-*-
  //        |         | |
  //       ---
  switch(plane_orientation) {
    case 0:
      plane_core_row = getRandomInt(1, BOARD_HEIGHT-1);
      plane_core_column = getRandomInt(2, BOARD_WIDTH-1);
      // Populate the tail
      agent_board[plane_core_row][plane_core_column-2] = 1;
      agent_board[plane_core_row-1][plane_core_column-2] = 1;
      agent_board[plane_core_row+1][plane_core_column-2] = 1;
      break;
    case 1:
      plane_core_row = getRandomInt(1, BOARD_HEIGHT-2);
      plane_core_column = getRandomInt(1, BOARD_WIDTH-1);
      // Populate the tail
      agent_board[plane_core_row+2][plane_core_column] = 1;
      agent_board[plane_core_row+2][plane_core_column+1] = 1;
      agent_board[plane_core_row+2][plane_core_column-1] = 1;
      break;
    case 2:
      plane_core_row = getRandomInt(1, BOARD_HEIGHT-1);
      plane_core_column = getRandomInt(1, BOARD_WIDTH-2);
      // Populate the tail
      agent_board[plane_core_row][plane_core_column+2] = 1;
      agent_board[plane_core_row-1][plane_core_column+2] = 1;
      agent_board[plane_core_row+1][plane_core_column+2] = 1;
      break;
    case 3:
      plane_core_row = getRandomInt(2, BOARD_HEIGHT-1);
      plane_core_column = getRandomInt(1, BOARD_WIDTH-1);
      // Populate the tail
      agent_board[plane_core_row-2][plane_core_column] = 1;
      agent_board[plane_core_row-2][plane_core_column+1] = 1;
      agent_board[plane_core_row-2][plane_core_column-1] = 1;
      break;
  }
  // Populate the cross
  agent_board[plane_core_row][plane_core_column] = 1;
  agent_board[plane_core_row+1][plane_core_column] = 1;
  agent_board[plane_core_row-1][plane_core_column] = 1;
  agent_board[plane_core_row][plane_core_column+1] = 1;
  agent_board[plane_core_row][plane_core_column-1] = 1;

  var dialog_state = {
    'agent_board'         :  agent_board,
    'user_board'          :  user_board,
    'total_hits_by_agent' :  0,
    'total_hits_by_user'  :  0,
    'agent_strike_row'    :  -1,
    'agent_strike_column' :  -1,
    'exposure'            :  0
  };
  return {
    'dialog_state' :  dialog_state
  };
}

/**
 * Generate the game agent's next strike position based on current game state.
 *
 * This function was implemented using a naive (and somewhat randomized)
 * neighbourhood search.
 *
 * Later a reinforcement learning model with much better performance was
 * introduced. So this function is pretty much obsolete and is only used as a
 * fallback when something really goes wrong with the RL prediction.
 *
 * However, combining neighbourhood search with reinforcement learning would
 * make the agent even stronger, especially at the last couple of steps. So I
 * might revisit this.
 *
 * @param {array} game_board User board
 * @param {number} total_hits_by_agent total successful hits made by the game
 *                 agent so far
 * @return {object} The object that contains a valid strike position
 *                  (row and column) or error message
 */
function NSPredictNextStrikePosition(game_board, total_hits_by_agent) {
  var unexplored_cells = [];
  var known_plane_cells = [];
  // Could have used datastore to optimize.
  // But since the latency isn't too bad, I'm not too worried about this.
  for (var i = 0; i < BOARD_HEIGHT; i++) {
    for (var j = 0; j < BOARD_WIDTH; j++) {
      if (game_board[i][j] == 1) {
        known_plane_cells.push([i, j]);
      }
      else if (game_board[i][j] == 0) {
        unexplored_cells.push([i, j]);
      }
    }
  }

  if (known_plane_cells.length != total_hits_by_agent) {
    // Something is wrong.
    return {
      'row'    : null,
      'column' : null,
      'error'  : 'Total user hits mismatched!'
    };
  }

  if (total_hits_by_agent == 0) {
    // Return a random unexplored cell.
    var cell = unexplored_cells[getRandomInt(0, unexplored_cells.length)];
    return {
      'row'    : cell[0],
      'column' : cell[1],
      'error'  : null
    };
  }

  // Rotate the array around a randomly chosen point.
  // Then do a straightforward search.
  var len = known_plane_cells.length;
  var pivot_point = getRandomInt(0, len);
  var search_array1 = known_plane_cells.slice(0, pivot_point);
  var search_array2 = known_plane_cells.slice(pivot_point, len);
  var search_array = search_array2.concat(search_array1);

  for (var i = 0; i < search_array.length; i++) {
    var row = search_array[i][0];
    var column = search_array[i][1];
    // Explore neighbourhood in 4 directions.
    // Could use more randomization.
    if (isUserCellEligible(game_board, row, column+1)) {
      return {
        'row'    : row,
        'column' : column+1,
        'error'  : null
      };
    }
    if (isUserCellEligible(game_board, row+1, column)) {
      return {
        'row'    : row+1,
        'column' : column,
        'error'  : null
      };
    }
    if (isUserCellEligible(game_board, row, column-1)) {
      return {
        'row'    : row,
        'column' : column-1,
        'error'  : null
      };
    }
    if (isUserCellEligible(game_board, row-1, column)) {
      return {
        'row'    : row-1,
        'column' : column,
        'error'  : null
      };
    }
  }

  return {
    'row'    : null,
    'column' : null,
    'error'  : 'I have exhausted all eligible cells to strike! '
  };
}


/**
 * Use trained reinforcement learning model to predict the agent's next
 * strike position.
 *
 * @param {array} game_board User board
 * @param {number} total_hits_by_agent total successful hits made by the game
 *                 agent so far
 * @return {object} The object that contains a valid strike position
 *                  (row and column) or error message
 */
function RLPredictNexStrikePosition(game_board, total_hits_by_agent) {
  var json = JSON.parse(jsonfile.readFileSync(NN_WEIGHT_FILE));
  var net = new convnetjs.Net();
  net.fromJSON(json);

  var flattened_game_board = [];
  var action_log = [];
  for (var i = 0; i < BOARD_HEIGHT; i++) {
    for (var j = 0; j< BOARD_WIDTH; j++) {
      flattened_game_board.push(game_board[i][j]);
    }
  }
  var flattened_game_board_vol = new convnetjs.Vol(flattened_game_board);
  var probs = net.forward(flattened_game_board_vol).w;
  var max = 0;
  var idx = -1;
  var hits_by_agent_cnt = 0;
  for (var i = 1; i < BOARD_SIZE; i++) {
    var x = Math.floor(i / BOARD_WIDTH);
    var y = i % BOARD_WIDTH;
    // For error checking purposes
    if (game_board[x][y] == 1) hits_by_agent_cnt++;
    // argmax
    if (probs[i] > max && game_board[x][y] == 0) {
      max = probs[i];
      idx = i;
    }
  }
  if (idx == -1) {
    // Something is wrong. Falling back to neighbourhood search ...
    return NSPredictNextStrikePosition(game_board, total_hits_by_agent);
  }
  if (total_hits_by_agent != hits_by_agent_cnt) {
    return {
      'row'    : null,
      'column' : null,
      'error'  : 'Total user hits mismatched!'
    };
  }
  return {
    'row'    : Math.floor(idx / BOARD_WIDTH),
    'column' : idx % BOARD_WIDTH,
    'error'  : null
  };
}

/**
 * Check if a particular strike position is eligible as the agent's next
 * strike position
 *
 * @param {array} game_board User board
 * @param {number} row Row # of candidate strike position
 * @param {number} col Column # of candidate strike position
 * @return {boolean} True if the position is eligible
 */
function isUserCellEligible(game_board, row, col) {
  if (row < 0 || row >= BOARD_HEIGHT) {
    return false;
  }
  if (col < 0 || col >= BOARD_WIDTH) {
    return false;
  }
  if (game_board[row][col]) {
    return false;
  }
  return true;
}

/**
 * Plane Strike game
 * Game rules:
 *            planestrike.wordpress.com
 *
 * @param {object} request JSON request
 * @param {object} response JSON response
 */

function planestrike(request, response) {
  var assistantRequest = request.body.originalRequest ?
                           request.body.originalRequest.data :
                           null;

  if (request.body.result) {
    var action = request.body.result.action;
    var intentName = request.body.result.metadata.intentName;
    var text_to_speech;
    switch (action) {
      case "START_PLANE_STRIKE":

        if (request.body.hasOwnProperty('originalRequest')) {
          var input_arg = request.body.originalRequest.data.inputs[0].arguments[0];

          if (input_arg != null && input_arg.name == "is_health_check") {
              text_to_speech = 'Hey Google, I\'m healthy';
              var msg = {
                  speech: text_to_speech,
                  data: {google: {expect_user_response: false}},
                  contextOut: []
              };
              response.send(msg);

              //dashbot.logOutgoing(request.body, msg);

              console.log('*****Google health checking ping received*****');
          }
        }
        else {
          var initial_state = initPlaneStrike();
          var dialog_state = initial_state.dialog_state;

          text_to_speech = 'Welcome to Plane Strike! If this is the '
            + 'first time you play this game, please visit '
            + 'plane strike dot word press dot com for '
            + 'instructions. Once you are ready, please '
            + 'let me know your first strike position by '
            + 'speaking out row and column number, for example,'
            + '2, 3.';
          var msg = {
            speech: text_to_speech,
            data: { google: { expect_user_response: true } },
            contextOut: [
              {
                name        : "user_strike_pos",
                lifespan    : 100,
                parameters  : dialog_state
              },
              {
                name        : "agent_hit_or_miss",
                lifespan    : 0
              }
            ]
          };
          response.send(msg);

          console.log('********Starting a new game!*********');
        }
        break;
      case "USER_CONFIRM_HIT_OR_MISS":
        // hit_or_miss: 'true' means it's a hit
        //              'false' means it's a miss
        var hit_or_miss = request.body.result.parameters.hit_or_miss;

        var dialog_state = null;
        for (var i = 0; i < request.body.result.contexts.length; i++) {
          if (request.body.result.contexts[i].name == 'agent_hit_or_miss') {
            dialog_state = request.body.result.contexts[i].parameters;
            break;
          }
        }
        if (!dialog_state) {
          console.log('*****No matching context in ' + action + '*****');
          var msg = {
            speech: 'Say that again?',
            data: { google: { expect_user_response: true } },
          };
          response.send(msg);
          //dashbot.logOutgoing(request.body, msg);
          //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
          break;
        }

        var agent_board = dialog_state.agent_board;
        var user_board = dialog_state.user_board;
        var total_hits_by_agent = dialog_state.total_hits_by_agent;
        var total_hits_by_user = dialog_state.total_hits_by_user;
        var agent_strike_row = dialog_state.agent_strike_row;
        var agent_strike_column = dialog_state.agent_strike_column;

        console.log('*** User confirms it\'s a hit? ' + hit_or_miss
          + ' ***');
        if (hit_or_miss == "yes") {
          // agent hit
          dialog_state.total_hits_by_agent += 1;
          console.log('*** total agent hits: '
            + dialog_state.total_hits_by_agent);
          if (user_board[agent_strike_row][agent_strike_column]) {
            console.log('INFO: Repeat shot is being made by the '
              + 'agent at ' + agent_strike_row + ' '
              + agent_strike_column);
          }
          user_board[agent_strike_row][agent_strike_column] = 1;
          if (dialog_state.total_hits_by_agent == PLANE_SIZE) {
            // agent wins
            text_to_speech = 'Awesome! I believe I have won the '
              + 'game by hitting all ' + PLANE_SIZE
              + ' parts of your plane. Thank you '
              + 'for playing!';
            var msg = {
              speech: text_to_speech,
              data: { google: { expect_user_response: false } },
              contextOut: [
                {
                  name        : "user_strike_pos",
                  lifespan    : 0
                },
                {
                  name        : "agent_hit_or_miss",
                  lifespan    : 0
                }
              ]
            };
            response.send(msg);
            //dashbot.logOutgoing(request.body, msg);
            //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
            break;
          }
          else {
            text_to_speech = 'Great! What\'s your next strike position? ';
            if (dialog_state.exposure < VERBOSE_LIMIT) {
              text_to_speech += 'Tell me the row and column number. '
                + 'For example, 3, 4';
              dialog_state.exposure += 1;
            }
            var msg = {
              speech: text_to_speech,
              data: { google: { expect_user_response: true } },
              contextOut: [
                {
                  name        : "user_strike_pos",
                  lifespan    : 100,
                  parameters  : dialog_state
                },
                {
                  name        : "agent_hit_or_miss",
                  lifespan    : 0
                }
              ]
            };
            response.send(msg);
            //dashbot.logOutgoing(request.body, msg);
            //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
            break;
          }
        }
        else {
          // agent missed
          user_board[agent_strike_row][agent_strike_column] = -1;
          text_to_speech = 'Too bad I missed! What\'s your next '
            + 'strike position? ';
          if (dialog_state.exposure < VERBOSE_LIMIT) {
            text_to_speech += 'Tell me the row '
              + 'and column number. For example, 5, 1';
              dialog_state.exposure += 1;
          }
          var msg = {
            speech: text_to_speech,
            data: { google: { expect_user_response: true } },
            contextOut: [
              {
                name        : "user_strike_pos",
                lifespan    : 100,
                parameters  : dialog_state
              },
              {
                name        : "agent_hit_or_miss",
                lifespan    : 0
              }
            ]
          };
          response.send(msg);
          //dashbot.logOutgoing(request.body, msg);
          //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
          break;
        }
      case "USER_PROVIDE_STRIKE_RC_POSITION":
      case "USER_PROVIDE_STRIKE_POSITION":
        console.log(request.body.result.parameters);
        if (request.body.result.parameters.user_strike_pos[0][0] &&
            request.body.result.parameters.user_strike_pos[0][1]) {
          // shift by 1
          var user_strike_row =
            parseInt(request.body.result.parameters.user_strike_pos[0][0]) - 1;
          // shift by 1
          var user_strike_column =
            parseInt(request.body.result.parameters.user_strike_pos[0][1]) - 1;
        }
        else if (request.body.result.parameters.row &&
                 request.body.result.parameters.column) {
          console.log(request.body.result.parameters);
          var user_strike_row = parseInt(request.body.result.parameters.row) - 1;
          var user_strike_column = parseInt(request.body.result.parameters.column) - 1;
        }
        else {
          console.log('*****ASR couldn\'t recognize the numbers*****');
          var msg = {
            speech: 'Could you say that again? For example, you could say 5,'
                    + ' 3 to attack row 5 column 3',
            data: { google: { expect_user_response: true } },
            contextOut: request.body.result.contexts
          };
          response.send(msg);
          //dashbot.logOutgoing(request.body, msg);
          //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
          break;
        }

        var dialog_state = null;
        for (var i = 0; i < request.body.result.contexts.length; i++) {
          if (request.body.result.contexts[i].name == 'user_strike_pos') {
            dialog_state = request.body.result.contexts[i].parameters;
            break;
          }
        }
        if (!dialog_state) {
          console.log('*****No matching context in ' + action + '*****');
          var msg = {
            speech: 'Say that again?',
            data: { google: { expect_user_response: true } },
          };
          response.send(msg);
          //dashbot.logOutgoing(request.body, msg);
          //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
          break;
        }

        if (user_strike_row > BOARD_HEIGHT - 1   ||
            user_strike_row < 0                  ||
            user_strike_column > BOARD_WIDTH - 1 ||
            user_strike_column < 0) {

          text_to_speech = 'I\'m sorry. Please give 2 numbers '
            + 'between 1 and 6 as your strike position.';
          var msg = {
            speech: text_to_speech,
            data: { google: { expect_user_response: true } },
            contextOut: [
              {
                name        : "user_strike_pos",
                lifespan    : 100,
                parameters  : dialog_state
              },
              {
                name        : "agent_hit_or_miss",
                lifespan    : 0
              }
            ]
          };
          response.send(msg);
          //dashbot.logOutgoing(request.body, msg);
          //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
          break;
        }

        var agent_board = dialog_state.agent_board;
        var user_board = dialog_state.user_board;
        var total_hits_by_agent = dialog_state.total_hits_by_agent;
        var total_hits_by_user = dialog_state.total_hits_by_user;

        if (agent_board[user_strike_row][user_strike_column] == 1) {
          // User hit.
          dialog_state.total_hits_by_user += 1;
          agent_board[user_strike_row][user_strike_column] = 2;
          console.log('*** total user hits: '
            + dialog_state.total_hits_by_user);
          if (dialog_state.total_hits_by_user == PLANE_SIZE) {
            text_to_speech = 'Congratulations! You have won '
              + 'the game!';
            var msg = {
              speech: text_to_speech,
              data: { google: { expect_user_response: false } },
              contextOut: [
                {
                  name        : "user_strike_pos",
                  lifespan    : 0
                },
                {
                  name        : "agent_hit_or_miss",
                  lifespan    : 0
                }
              ]
            };
            response.send(msg);
            //dashbot.logOutgoing(request.body, msg);
            //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
            break;
          }
          text_to_speech = 'Oh, row ' + (user_strike_row+1)
            + ' column ' + (user_strike_column+1)
            + ' was a hit! Now it\'s my turn. Strike '
            + 'at ';
        }
        else if (agent_board[user_strike_row][user_strike_column] == 2) {
          text_to_speech = 'You\'ve just made a repeat shot! Row '
            + (user_strike_row+1) + ' column '
            + (user_strike_column+1) + ' was a hit. '
            + 'Now it\'s my turn. Strike at ';
        }
        else if (agent_board[user_strike_row][user_strike_column] == -1) {
          text_to_speech = 'You\'ve just made a repeat shot! Row '
            + (user_strike_row+1) + ' column '
            + (user_strike_column+1) + ' was a miss. '
            + 'Now it\'s my turn. Strike at ';
        }
        else {
          // User missed.
          agent_board[user_strike_row][user_strike_column] = -1;
          text_to_speech = 'Unfortunately row ' +  (user_strike_row+1)
            + ' column ' + (user_strike_column+1)
            + ' was a miss! Now it\'s my turn. Strike at ';
        }
        var nextStrikePosition = RLPredictNexStrikePosition(user_board,
          total_hits_by_agent);
        if (nextStrikePosition.error) {
          // the game is screwed up
          text_to_speech = nextStrikePosition.error;
          text_to_speech += 'Unfortunately the internal state of the '
            + 'game is messed up. Please restart the '
            + 'game!';
          var msg = {
            speech: text_to_speech,
            data: { google: { expect_user_response: false } },
            contextOut: [
              {
                name        : "user_strike_pos",
                lifespan    : 0
              },
              {
                name        : "agent_hit_or_miss",
                lifespan    : 0
              }
            ]
          };
          response.send(msg);
          //dashbot.logOutgoing(request.body, msg);
          //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
          break;
        }
        var agent_strike_row = nextStrikePosition.row;
        var agent_strike_column = nextStrikePosition.column;
        text_to_speech += 'row ' + (agent_strike_row + 1) + ' column '
          + (agent_strike_column + 1)
          + '. Did I hit your plane?';
        console.log((agent_strike_row+1) + ', ' + (agent_strike_column+1));
        dialog_state.agent_strike_row = agent_strike_row;
        dialog_state.agent_strike_column = agent_strike_column;
        var msg = {
          speech: text_to_speech,
          data: { google: { expect_user_response: true } },
          contextOut: [
            {
              name        : "user_strike_pos",
              lifespan    : 0
            },
            {
              name        : "agent_hit_or_miss",
              lifespan    : 100,
              parameters  : dialog_state
            }
          ]
        };
        response.send(msg);
        //dashbot.logOutgoing(request.body, msg);
        //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
        break;
      case "USER_STOP_GAME_1":
      case "USER_STOP_GAME_2":
        console.log('*****User stops the game*****');
        text_to_speech = 'Stopping plane strike game. Thank you for '
          + 'playing the game!';
        var msg = {
          speech: text_to_speech,
          data: { google: { expect_user_response: false } },
          contextOut: [
            {
              name        : "user_strike_pos",
              lifespan    : 0
            },
            {
              name        : "agent_hit_or_miss",
              lifespan    : 0
            }
          ]
        };
        response.send(msg);
        //dashbot.logOutgoing(request.body, msg);
        //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
        break;
      default:
        console.log('*****Unsupported action*****');
        console.log(action);

        var resolvedQuery = request.body.
                                    result.
                                    resolvedQuery;
        if (resolvedQuery != null) {
          text_to_speech = 'I heard: ' + resolvedQuery +
                             '. Could you say that again?';
        }
        else {
          text_to_speech = 'Unsupported action. Could you say that again?';
        }

        console.log(text_to_speech);

        var msg = {
          speech: text_to_speech,
          data: { google: { expect_user_response: true } },
          contextOut: request.body.result.contexts
        };
        response.send(msg);
        //dashbot.logOutgoing(request.body, msg);
        //VoiceInsights.track(intentName, assistantRequest, text_to_speech);
        break;
    }
  }
}
