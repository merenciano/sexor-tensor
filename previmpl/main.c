#include "nn.c"

struct MLP mlp;

void LoadTrainingData(double **pix, double **label)
{
	FILE *f = fopen("mnist/train-digits", "rb");
	*pix = malloc(60000 * 13 * 13 * sizeof(double));
	if (!*pix)
	{
		printf("Bad alloc Pix\n");
	}
	*label = malloc(60000 * sizeof(double));
	if (!*label)
	{
		printf("Bad alloc Label\n");
	}
	fseek(f, 16, SEEK_SET);
	for (int i = 0; i < 60000; ++i)
	{
		char img[26][26];
		double *img1 = *pix + (13 * 13 * i);
		fseek(f, 28, SEEK_CUR);
		for (int j = 0; j < 26; ++j)
		{
			fseek(f, 2, SEEK_CUR);
			fread(img[j], 26, 1, f);
			fseek(f, 2, SEEK_CUR);
		}
		fseek(f, 28, SEEK_CUR);

		for (int y = 0; y < 26; y += 2)
		{
			for (int x = 0; x < 26; x += 2)
			{
				double val = (double)img[y][x] + (double)img[y][x+1];
				val += (double)img[y+1][x] + (double)img[y+1][x+1];
				val /= 510.0;
				val -= 1.0;
				img1[(y/2) * 13 + (x/2)] = val;
			}
		}
	}
	fclose(f);

	f = fopen("mnist/train-labels", "rb");
	char labs[60000];
	fseek(f, 8, SEEK_SET);
	fread(labs, 60000, 1, f);
	for (int i = 0; i < 60000; ++i)
	{
		(*label)[i] = (double)labs[i];
	}
	fclose(f);
}

void SaveParams(double *values, size_t count)
{

}

int main()
{
	v = malloc(MAX_UNITS * sizeof(double));
	g = malloc(MAX_UNITS * sizeof(double));
	c = malloc(MAX_UNITS * 2U * sizeof(int));
	op = malloc(MAX_UNITS);
	topo = malloc(MAX_UNITS * sizeof(int));
	topo_visited = malloc((MAX_UNITS / 64) * sizeof(uint64_t));


	int layers[3] = {13*13, 13, 1};
	double *data;
	double *pred;
	LoadTrainingData(&data, &pred);

	InitMLP(&mlp, layers, 2);

	Learn(&mlp, data, 10, pred, 0.03, 0.001);

	//MLP_SaveParams(&mlp, "params.bin");

	return 0;
}
