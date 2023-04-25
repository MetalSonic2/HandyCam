/*
<script>
    function submitted(e){
        e.preventDefault();
        
        let namee = documenr.forms["id"]["name"].value;
        sessionStorage.setItem("name", namee);
    }
</script>
*/

let questions = [
    {
        id: 1,
        question: "In ASL, show letter A:"
    },
    {
        id: 2,
        question: "In ASL, show letter B:"
    },
    {
        id: 3,
        question: "In ASL, show letter C:"
    },
    {
        id: 4,
        question: "In ASL, show letter D:"
    },
    {
        id: 5,
        question: "In ASL, show letter E:"
    },
    {
        id: 6,
        question: "In ASL, show letter F:"
    },
    {
        id: 7,
        question: "In ASL, show letter G:"
    },
    {
        id: 8,
        question: "In ASL, show letter H:"
    },
    {
        id: 9,
        question: "In ASL, show letter I:"
    },
    {
        id: 10,
        question: "In ASL, show letter J:"
    },
    {
        id: 11,
        question: "In ASL, show letter K:"
    },
    {
        id: 12,
        question: "In ASL, show letter L:"
    },
    {
        id: 13,
        question: "In ASL, show letter M:"
    },
    {
        id: 14,
        question: "In ASL, show letter N:"
    },
    {
        id: 15,
        question: "In ASL, show letter O:"
    },
    {
        id: 16,
        question: "In ASL, show letter P:"
    },
    {
        id: 17,
        question: "In ASL, show letter Q:"
    },
    {
        id: 18,
        question: "In ASL, show letter R:"
    },
    {
        id: 19,
        question: "In ASL, show letter S:"
    },
    {
        id: 20,
        question: "In ASL, show letter T:"
    },
    {
        id: 21,
        question: "In ASL, show letter U:"
    },
    {
        id: 22,
        question: "In ASL, show letter V:"
    },
    {
        id: 23,
        question: "In ASL, show letter W:"
    },
    {
        id: 24,
        question: "In ASL, show letter X:"
    },
    {
        id: 25,
        question: "In ASL, show letter Y:"
    },
    {
        id: 26,
        question: "In ASL, show letter Z:"
    },
];

let question_count = 0;
let points = 0;

window.onload = function () {
    show(question_count);
};

function show(count) {
    let question = document.getElementById("questions");

    question.innerHTML = `<h2>Q${count + 1}. ${questions[count].question}</h2>`;

    toggleActive();
}

function next() {
    if (question_count == questions.length - 1) {
        location.href = "final.html";
    }
    console.log(question_count);
    //let user_answer = document.querySelector()

    let user_answer = 0;

    if (user_answer > 0.79) {
        points += 10;
        sessionStorage.setItem["points", points];
    }
    console.log(points);

    question_count++;
    show(question_count);
}

