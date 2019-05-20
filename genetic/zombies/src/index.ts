
function mod(x: number, n: number): number {
  return (x % n + n) % n;
}

export interface Zombie {
  sprite: PIXI.Sprite;
  vector: number[];
  follow: boolean;
  speed: number;
  perception: number;
}

export interface Human {
  sprite: PIXI.Sprite;
  vector: number[];
  followdist: number;
  speed: number;
  perception: number;
  ammunition: number;
  shot_accuracy: number;
  genome: number[];
  lifeduration: number;
}

export interface Bullet {
  sprite: PIXI.Sprite;
}

export interface WorldConf {
  [key:string]: any,
  nbzombie: number;
  nbhuman: number;
}

export interface Phenotype {
  speed: number,
  perception: number,
  shot_accuracy: number,
  ammunition: number
}

export interface Position {
  y: number;
  x: number;
}

class GeneticZombie {

  private canvasId: string;
  protected app: PIXI.Application = null; // The pixi app.
  protected sprite: PIXI.Sprite = null;
  protected zombies: Zombie[] = [];
  protected bullets: Bullet[] = [];
  protected humans: Human[] = [];
  protected humansToConvert: string[] = [];
  protected zombieToKill: string[] = [];
  protected sounds: Position[] = [];
  protected isplaying: boolean = true;
  protected population: Human[] = [];
  protected last_population: Human[] = [];
  protected stepnb: number = 0;
  protected callback: any = null;

  protected width: number = 1200;
  protected height: number = 800;

  protected config: WorldConf = {
      nbzombie: 50,
      nbhuman: 50,
      nbbullets: 10,
  };

  constructor(canvasId: string, config: WorldConf) {
      if (!canvasId){
          console.error("You must specify the canvasId");
      }

      for (let key in this.config){
        if (config && key in config){
          this.config[key] = config[key];
        }
      }

      this.canvasId = canvasId;//Create a Pixi Application
      this.app = new PIXI.Application({
        width: this.width,
        height: this.height,
        backgroundColor: 0x002628
      });
      //Add the canvas that Pixi automatically created for you to the HTML document
      document.getElementById(this.canvasId).appendChild(this.app.view);

      this.app.ticker.add(delta => {
        if (this.callback){
          this.callback(this, delta);
        }
      });
    }

    protected vectorrand(vector: number []){
       let a = Math.random();
       let b = 1 - a;
       vector[0] = vector[0] * a;
       vector[1] = vector[1] * b;
       return vector;
    }

    public run(){
      this.isplaying = true;
    }

    public stop(){
      this.isplaying = false;
    }

    /*
      Step in the environment
    */
    public step(delta: number){
        this.stepnb += 1;

        // Track the human that are going to be catch by zombies
        this.humansToConvert = [];
        // Step all the zombies
        let n_zombies: Zombie[] = [];
        for (let i in this.zombies) {
          this.stepZombie(this.zombies[i], i);
          if (this.zombieToKill.indexOf(i) == -1){
            n_zombies.push(this.zombies[i]);
          }
        }
        this.zombies = n_zombies;
        if (this.zombies.length == 0){
          let zombie = this.createZombie();
        }
        // Track the zombie that are going to be kill
        this.zombieToKill = [];
        // Step all the humans
        this.sounds = [];
        let n_humans: Human[] = [];
        for (let i in this.humans) {
          this.stepHuman(this.humans[i], i);
          if (this.humansToConvert.indexOf(i) == -1){
            this.humans[i].lifeduration = this.stepnb;
            n_humans.push(this.humans[i])
          }
          else {
            let zombie = this.createZombie();
            zombie.sprite.position.x = this.humans[i].sprite.position.x;
            zombie.sprite.position.y = this.humans[i].sprite.position.y;
            this.humans[i].sprite.visible = false;
          }
        }
        this.humans = n_humans;
    }

    /*
      Move a particular Zombie
    */
    public stepZombie(z: Zombie, hid: string){
        for (let s in this.sounds){
          let sound = this.sounds[s];
          let dist = Math.sqrt((z.sprite.position.x-sound.x)**2 + (z.sprite.position.y-sound.y)**2);
          if (dist < 200){
            z.vector[0] = sound.x <= z.sprite.position.x ? -1 : 1;
            z.vector[1] = sound.y <= z.sprite.position.y ? -1 : 1;
          }
        }
        for (let i in this.humans) {
          let h = this.humans[i];

          let dist = Math.sqrt((h.sprite.position.x-z.sprite.position.x)**2 + (h.sprite.position.y-z.sprite.position.y)**2);

          if (dist < 10){
            if (this.humansToConvert.indexOf(i) == -1){
                this.humansToConvert.push(i);
                this.app.stage.removeChild(h.sprite);
            }
          }
          // The speed of the human increase the perception of the zombie
          // because of the sound
          let factornoise = h.speed > 1.0 ? (h.speed-1.0)*50 : 0.0;
          if (dist < z.perception * factornoise){
             z.vector[0] = h.sprite.position.x <= z.sprite.position.x ? -1 : 1;
             z.vector[1] = h.sprite.position.y <= z.sprite.position.y ? -1 : 1;
             //z.vector = this.vectorrand(z.vector);
          }
        }
        z.sprite.position.x = mod(z.sprite.position.x + (z.speed*z.vector[0]) , this.width);
        z.sprite.position.y = mod(z.sprite.position.y + (z.speed*z.vector[1]),  this.height);
    }

    /*
      Move a particular human
    */
    public stepHuman(h: Human, hid: string): void {

      h.followdist = 10000;

      let n_bullets: Bullet[] = [];
      let bullet_to_add = 0;
      for (let b in this.bullets) {
        let bullet = this.bullets[b];
        let dist = Math.sqrt((h.sprite.position.x-bullet.sprite.position.x)**2 + (h.sprite.position.y-bullet.sprite.position.y)**2);
        if (dist < h.perception){
          h.vector[0] = h.sprite.position.x <= bullet.sprite.position.x ? 1 : -1;
          h.vector[1] = h.sprite.position.y <= bullet.sprite.position.y ? 1 : -1;
          h.vector = this.vectorrand(h.vector);
        }
        // Take the bullets
        if (dist < 10){
          bullet.sprite.visible = false;
          this.app.stage.removeChild(bullet.sprite);
          h.ammunition += 5;
          bullet_to_add += 1;
        } else{
          n_bullets.push(bullet);
        }
      }
      this.bullets = n_bullets;
      for (let b=0; b < bullet_to_add; b++){
        this.createBullet();
      }

      for (let i in this.zombies) {
        let z = this.zombies[i];

        let dist = Math.sqrt((h.sprite.position.x-z.sprite.position.x)**2 + (h.sprite.position.y-z.sprite.position.y)**2);

        if (dist < h.perception && h.followdist+10 > dist){
          if (h.ammunition > 0){ // Try to shot the Zombie
              h.ammunition = h.ammunition - 1;
              this.sounds.push({
                y: h.sprite.position.y,
                x: h.sprite.position.x
              });
              if (Math.random() < h.shot_accuracy){
                this.zombieToKill.push(i);
                this.app.stage.removeChild(z.sprite);
              }
          }
           h.vector[0] = h.sprite.position.x <= z.sprite.position.x ? -1 : 1;
           h.vector[1] = h.sprite.position.y <= z.sprite.position.y ? -1 : 1;
           h.vector = this.vectorrand(h.vector);
           h.followdist = dist;
        }
      }

      h.sprite.position.x = mod(h.sprite.position.x + (h.speed*h.vector[0]), this.width);
      h.sprite.position.y = mod(h.sprite.position.y + (h.speed*h.vector[1]), this.height);
    }

    public emptyHumans(){
      for (let h in this.humans){
        this.humans[h].sprite.visible = false;
        this.app.stage.removeChild(this.humans[h].sprite);
      }
      this.humans = [];
      this.last_population = this.population;
      this.population = [];
    }

    public emptyZomvies(){
      for (let z in this.zombies){
        this.zombies[z].sprite.visible = false;
        this.app.stage.removeChild(this.zombies[z].sprite);
      }
      this.zombies = [];
    }

    public createZombies(){
      console.log("create zombies", this.config.nbzombie - this.zombies.length);
      for (let i = 0; i < this.config.nbzombie - this.zombies.length; i++) {
        this.createZombie();
      }
    }

    public reset() {
      this.stepnb = 0;

      for (let b in this.bullets){
        this.bullets[b].sprite.visible = false;
        this.app.stage.removeChild(this.bullets[b].sprite);
      }
      this.bullets = [];
      for (let z in this.zombies){
        this.zombies[z].sprite.visible = false;
        this.app.stage.removeChild(this.zombies[z].sprite);
      }
      this.zombies = [];
      for (let h in this.humans){
        this.humans[h].sprite.visible = false;
        this.app.stage.removeChild(this.humans[h].sprite);
      }
      this.humans = [];
      this.population = [];
      for (let i = 0; i < this.config.nbbullets; i++) {
        this.createBullet();
      }
      for (let i = 0; i < this.config.nbzombie; i++) {
        this.createZombie();
      }
    }

    public init(oninit: any, callback: any): void {
      PIXI.loader
      .add("/public/images/boy.png")
      .add("/public/images/girl.png")
      .add("/public/images/zombie.png")
      .add("/public/images/bullet.png")
      .load(() => {

          for (let i = 0; i < this.config.nbbullets; i++) {
            this.createBullet();
          }

          for (let i = 0; i < this.config.nbzombie; i++) {
            this.createZombie();
          }
          //for (let i = 0; i < this.config.nbhuman; i++) {
          //  this.createHuman();
          //}

          if (oninit){
            oninit(this);
          }

          this.callback = callback;

      });
    }

    /**
    * Create a new Zombie
    */
    public createZombie(): Zombie {
      let sprite = new PIXI.Sprite(PIXI.loader.resources["/public/images/zombie.png"].texture);
      sprite.scale.x = 0.05;
      sprite.scale.y = 0.05;
      // Position of the Zombie
      sprite.position.x = Math.floor(Math.random() * this.width);
      sprite.position.y = Math.floor(Math.random() * this.height);
      this.app.stage.addChild(sprite);

      let vector: number[] = [
        Math.random() > 0.75 ? -1 : 1,
        Math.random() > 0.25 ? -1 : 1
      ]
      vector = this.vectorrand(vector);

      let zombie: Zombie = {
        sprite: sprite,
        vector: vector,
        follow: false,
        speed: 0.8,
        perception: 30.0
      }
      this.zombies.push(zombie);
      return zombie;
    }


    /**
    * Create a Bullet
    */
    public createBullet(): Bullet {
      let sprite = new PIXI.Sprite(PIXI.loader.resources["/public/images/bullet.png"].texture);
      sprite.scale.x = 0.03;
      sprite.scale.y = 0.03;
      // Position of the Zombie
      sprite.position.x = Math.floor(Math.random() * this.width);
      sprite.position.y = Math.floor(Math.random() * this.height);
      this.app.stage.addChild(sprite);

      let bullet: Bullet = {
        sprite: sprite
      }

      this.bullets.push(bullet);
      return bullet;
    }

    /*
      Genome to phenotype
    */
    protected genomeToPhenotype(genome: number[]): Phenotype {
      let phenotype: Phenotype = {
        speed: 1.0 + (0.4*genome[0]-0.2),
        perception: 50 + (300*genome[1]-150),
        shot_accuracy: 0.3 + (0.6*genome[2]-0.3),
        ammunition: 4 + (20*genome[3]-10),
      };
      return phenotype;
    }

    /**
    * Create a new Human
    */
    public createHuman(genome: number[]): void {
      let phenotype = this.genomeToPhenotype(genome);
      let sprite = new PIXI.Sprite(PIXI.loader.resources[
        Math.random() > 0.5 ? "/public/images/boy.png" : "/public/images/girl.png"
      ].texture);
      sprite.scale.x = 0.05;
      sprite.scale.y = 0.05;
      // Position of the Zombie
      sprite.position.x = Math.floor(Math.random() * this.width);
      sprite.position.y = Math.floor(Math.random() * this.height);
      this.app.stage.addChild(sprite);

      let vector: number[] = [
        Math.random() > 0.5 ? -1 : 1,
        Math.random() > 0.5 ? -1 : 1
      ]
      vector = this.vectorrand(vector);

      let n_human = {
        sprite: sprite,
        vector: vector,
        followdist: 10000,
        speed: phenotype.speed,
        perception: phenotype.perception,
        ammunition: phenotype.ammunition,
        shot_accuracy: phenotype.shot_accuracy,
        genome: genome,
        lifeduration: 0
      };
      this.humans.push(n_human);
      this.population.push(n_human);
    }

}

const geneticzombie = {
  env: GeneticZombie
}

export default geneticzombie;
