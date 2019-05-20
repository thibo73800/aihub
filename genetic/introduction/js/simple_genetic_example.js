function individual_to_string(individual){
  let st = "";
  for (let i=0; i<8; i++){
    st += individual[i].toString();
  }
  return st;
}

function display_pop(pop){
  // Display each individual
  for (let p=0; p<8; p++){
    // Display the individual on the screen
    let st = individual_to_string(pop[p]);
    document.getElementById("ind_" + (p + 1)).innerHTML = st;
  }
}

function display_subpop(subpop){
  // Display each individual
  for (let p=0; p<4; p++){
    // Display the individual on the screen
    let st = individual_to_string(subpop[p]);
    document.getElementById("sel_" + (p + 1)).innerHTML = st;
  }
}

function compute_fitness(individual){
  score = 0;
  for (let i=0; i<8; i++){
    score += individual[i];
  }
  return score;
}

function cross_over(i1, i2){
  let individual = [
    i1[0],
    i1[1],
    i1[2],
    i1[3],
    i2[4],
    i2[5],
    i2[6],
    i2[7]
  ];
  return individual;
}

function generate_individual(){
  let individual = [];
  for (let i=0; i<8; i++){
    individual.push(Math.random() > 0.5 ? 1 : 0);
  }
  return individual;
}

function generate_population(){
  // Init empty population
  let pop = []
  for (let p=0; p<8; p++){
    // Randomly generate one individual with random genes
    pop.push(generate_individual());
  }
  return pop;
}

let pop = generate_population();
console.log(pop);
display_pop(pop);
//var n_pop = [];

document.getElementById("sort").addEventListener("click", () => {
  // Sort function
  pop.sort((i1, i2) => {
    return compute_fitness(i2) - compute_fitness(i1);
  });
  console.log(pop);
  display_pop(pop);
});

document.getElementById("selection").addEventListener("click", () => {
  n_pop = [];
  n_pop.push(pop[0]);
  n_pop.push(cross_over(pop[0], pop[1]));
  n_pop.push(cross_over(pop[0], pop[2]));
  n_pop.push(cross_over(pop[1], pop[2]));
  display_subpop(n_pop);
});

document.getElementById("mutate").addEventListener("click", () => {
  for (let i=0; i<4; i++){
    let individual = n_pop[i];
    let gene_to_mutate = Math.floor(Math.random()*individual.length);
    if (individual[gene_to_mutate] == 0) {
      individual[gene_to_mutate] = 1;
    }
    else {
      individual[gene_to_mutate] = 0;
    }
    display_subpop(n_pop);
  }
});

document.getElementById("fillpop").addEventListener("click", () => {
  pop = []
  pop.push(n_pop[0]);
  pop.push(n_pop[1]);
  pop.push(n_pop[2]);
  pop.push(n_pop[3]);

  pop.push(generate_individual());
  pop.push(generate_individual());
  pop.push(generate_individual());
  pop.push(generate_individual());

  display_pop(pop);

  document.getElementById("sel_1").innerHTML = "";
  document.getElementById("sel_2").innerHTML = "";
  document.getElementById("sel_3").innerHTML = "";
  document.getElementById("sel_4").innerHTML = "";
});
