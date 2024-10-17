const fetch = require('node-fetch');

// Function to log in and fetch evaluations
async function loginAndFetchEvaluations(username, password) {
    const loginUrl = 'https://eventretrieval.one/api/v2/login';
    const evaluationUrl = 'https://eventretrieval.one/api/v2/client/evaluation/list';
    let sessionId, evaluationId;

    try {
        // Step 1: Log in and get session ID
        const loginResponse = await fetch(loginUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        if (!loginResponse.ok) {
            throw new Error(`Login error: ${loginResponse.status} ${loginResponse.statusText}`);
        }

        const loginData = await loginResponse.json();
        sessionId = loginData.sessionId;
        console.log('Login successful:', loginData);
        console.log('Session ID:', sessionId);

        // Step 2: Get Evaluation IDs
        const evaluationResponse = await fetch(`${evaluationUrl}?session=${sessionId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!evaluationResponse.ok) {
            throw new Error(`Evaluation fetch error: ${evaluationResponse.status} ${evaluationResponse.statusText}`);
        }

        const evaluations = await evaluationResponse.json();
        console.log('Evaluations:', evaluations);

        // Get the first evaluation ID
        if (evaluations.length > 0) {
            evaluationId = evaluations[0].id; // Assuming you want the first evaluation
            console.log(`Selected Evaluation ID: ${evaluationId}`);
        } else {
            throw new Error('No evaluations found.');
        }

    } catch (error) {
        console.error('Operation failed:', error);
    }

    return { sessionId, evaluationId };
}

// Function to submit QA answers
async function submitQA(sessionId, evaluationId, answer, videoId, time) {
    const submitUrl = `https://eventretrieval.one/api/v2/submit/${evaluationId}`;
    
    const formattedAnswer = `${answer}-${videoId}-${time}`;
    
    const body = {
        session: sessionId,
        answerSets: [{
            answers: [{
                text: formattedAnswer
            }]
        }]
    };

    try {
        const response = await fetch(submitUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            throw new Error(`Submit QA error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Submit QA result:', result);
    } catch (error) {
        console.error('Submission failed:', error);
    }
}

// Function to submit KIS answers
async function submitKIS(sessionId, evaluationId, videoId, time) {
    const submitUrl = `https://eventretrieval.one/api/v2/submit/${evaluationId}`;

    const body = {
        session: sessionId,
        answerSets: [{
            answers: [{
                mediaItemName: videoId,
                start: time,
                end: time // Use the same time for start and end
            }]
        }]
    };

    try {
        const response = await fetch(submitUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            throw new Error(`Submit KIS error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Submit KIS result:', result);
    } catch (error) {
        console.error('Submission failed:', error);
    }
}

// Event listener for getting session ID and evaluation ID
document.getElementById('get-session-button').addEventListener('click', async () => {
    const { sessionId, evaluationId } = await loginAndFetchEvaluations('team70', 'EwetvYSbAZ');

    // Enable the submit buttons after fetching evaluations
    document.getElementById('submit-qa-button').disabled = false;
    document.getElementById('submit-kis-button').disabled = false;

    // Store the session ID and evaluation ID for later use
    window.sessionId = sessionId;
    window.evaluationId = evaluationId;
});

// Event listener for the QA submission button
document.getElementById('submit-qa-button').addEventListener('click', () => {
    const answer = prompt("Enter your QA answer:");
    const videoId = prompt("Enter Video ID:");
    const time = prompt("Enter Time (ms):");

    if (answer && videoId && time) {
        submitQA(window.sessionId, window.evaluationId, answer, videoId, time);
    }
});

// Event listener for the KIS submission button
document.getElementById('submit-kis-button').addEventListener('click', () => {
    const videoId = prompt("Enter Video ID:");
    const time = prompt("Enter Time (ms):"); // Prompt only for one time input

    if (videoId && time) {
        submitKIS(window.sessionId, window.evaluationId, videoId, time);
    }
});
