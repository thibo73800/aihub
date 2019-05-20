/*
  Create the initial population in the environment
*/
function createPopulation(env) {
  // Create individuals with one specific or random genome
  for (let i = 0; i < 15; i++){
    //env.createHuman([Math.random(), Math.random(), Math.random(), Math.random()]);
    env.createHuman([0.5, 0.5, 0.5, 0.5]);
  }
  // Create individuals with one specific or random genome
  for (let i = 0; i < 15; i++){
    //env.createHuman([Math.random(), Math.random(), Math.random(), Math.random()]);
    env.createHuman([0.5, 0.5, 0.5, 0.5]);
  }
}

/*
  Method used to select amoung the last population the best
  individuals to create a new population
*/
function selection(env){
  let genomed = [0, 0, 0, 0, 0];

  // Compute the total fitness of all individuals
  let total_fitness = 0;
  for (let p in env.last_population){
    total_fitness += env.last_population[p].lifeduration;
  }

  // Sort individual by their best fitness
  env.last_population.sort((i1, i2) => {
      i1.prob = i1.lifeduration / total_fitness;
      i2.prob = i2.lifeduration / total_fitness;
      return i2.lifeduration - i1.lifeduration;
  });

  for (let i = 0; i<30; i++){

    let rd = Math.random();
    let lastprop = 0;

    for (let p in env.last_population){
      let h = env.last_population[p];
      if (rd >= lastprop && rd < h.prob + lastprop){
        //if (h.genome[3] == 0.75){
        genomed = update_genome_to_display(genomed, h.genome);
        //}
        env.createHuman(env.last_population[p].genome);
        break;
      }
      lastprop += env.last_population[p].prob;
    }

  }

  display_genome(genomed);
}

function step(env, delta){
    // Retrieve the desired speed
    let speed = document.getElementById("speed").value;
    // Step in the environment with the given speed
    for (let i = 0; i<speed; i++){
      env.step();
    }
    // If all Humans died or all zombies died
    if (env.humans.length <= 1 || env.zombies.length <= 1){
        env.emptyHumans(); // Remove all Human left
        env.emptyZomvies(); // Remove all zombies left
        selection(env); // Select the new population
        env.createZombies(); // Create the zombies
        //document.getElementById("speed").value = 1;
    }
    // Step again a the next animation frame
    window.requestAnimationFrame(() => {
      step(env);
    });
}

function init(env){
  env.reset();
  createPopulation(env);
  step(env);
}

var env = new geneticzombie.env("canvas_container", {nbbullets : 10});
env.init(init, null);
