function update_genome_to_display(genomed, genome){
  genomed = [
    genomed[0]+1,
    genomed[1]+genome[0],
    genomed[2]+genome[1],
    genomed[3]+genome[2],
    genomed[4]+genome[3]
  ]
  return genomed;
}

function display_genome(genome){
  console.log(chart_genome);
  genome = [
    genome[0]/30, // Scale between 0 and 1
    genome[1]/genome[0], // Take the mean
    genome[2]/genome[0], // Take the mean
    genome[3]/genome[0], // Take the mean
    genome[4]/genome[0] // Take the mean
  ]
  chart_genome.data.datasets.forEach((dataset) => {
    dataset.data = genome;
  });
  chart_genome.update();
}

var ctx = document.getElementById('genomeChart').getContext('2d');
var chart_genome = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['Size', 'Gen1 (Speed)', 'Gen2 (Perception)', 'Gen3 (Accuracy)', 'Gen4 (bullets)'],
        datasets: [{
            label: 'Genome',
            labelColor: 'rgba(0, 99, 132, 0.2)',
            data: [],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
      responsive:false,
      maintainAspectRatio: false,
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true
                }
            }]
        }
    }
});
