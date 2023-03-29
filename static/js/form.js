// const btnDisplayOnFormChange = () => {
const formInput = document.getElementById('file');
const btn = document.getElementById('button-submit');
formInput.addEventListener('change', () => {

    btn.style.display = 'block';
});
// }
