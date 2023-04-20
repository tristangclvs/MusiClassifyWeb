// const btnDisplayOnFormChange = () => {

const formInput = document.getElementById('file');
const btn = document.getElementById('button-submit');
const main = document.getElementById('main');

formInput.addEventListener('change', () => {
        console.log('form changed');
        const selectedFile = formInput.files[0];

        // Create a blob that we can use as an src for our audio element
        const urlObj = URL.createObjectURL(selectedFile);

        // Create an audio element
        const audio = document.createElement("audio");

        // Clean up the URL Object after we are done with it
        audio.addEventListener("load", () => {
            URL.revokeObjectURL(urlObj);
        });

        // Append the audio element
        main.appendChild(audio);

        // Allow us to control the audio
        audio.controls = "true";

        // Set the src and start loading the audio from the file
        audio.src = urlObj;

        btn.style.display = 'block';
    }
);
