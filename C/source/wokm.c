/* 
    WOKM Implementation

    Author: Argenis Aldair Aroche Villarruel
    Date: March 2013

    This is the implementation used for the comparison in the paper: 
    Study of Overlapping Clustering Algorithms Based on Kmeans through FBcubed Metric
    https://doi.org/10.1007/978-3-319-07491-7_12

    Implementation based on: 
    Two Variants of the O KM for Overlapping Clustering
    https://doi.org/10.1007/978-3-642-00580-0_9

    Disclaimer:
    This code is provided "as is" without warranty of any kind, express or
    implied, including but not limited to the warranties of merchantability,
    fitness for a particular purpose, and noninfringement. In no event shall
    the authors or copyright holders be liable for any claim, damages, or other
    liability, whether in an action of contract, tort, or otherwise, arising
    from, out of, or in connection with the software or the use or other
    dealings in the software.

    Usage:
    - This code is for educational and research purposes only.
    - The author is not responsible for any malfunction, damage, or loss of
    data caused by the use of this code.

    Please acknowledge the author if you use or modify this code for your
    own projects or research.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define BUFFER 2048000
#define MISSING -1

typedef struct {
    int    d;  /* Discrete value (possibly discretized) (-1='?'=MISSING) */
    float  f;  /* Floating-point (continuous) value */
 } Value;
 
 typedef struct{
     int *A;
     int *W;
     int size;
     int flag;
 }AssignList;

typedef struct{	
    int index;
    int class;
    int nAttrib;
    int *R;
    float *weight;
    float *auxWeight;
    AssignList *list;
    AssignList *oldList;
    Value *Attrib;
}Instance;

typedef struct{
    int size;
    Instance representative;
    int *instances;
}Cluster;

typedef struct{
    int nFeatures;
    int nClasses;
    int size;
    int *type;
}Header;

typedef struct{
    int index;
    float val;
}Range;



//Var
Header *head;
Instance *training, *imageX, *oldImage;
Cluster *clusters;
int nAttrib = 0, nInst = 0, nClass = 0, iterations = 0;
time_t start, end;
//Functions


/*File and Utils*/

void printInstance(Instance *inst){
    int i;
    for(i = 0; i < nAttrib; i++){
        if(inst->Attrib[i].d == MISSING){
            printf("?, ");
        }
        else if(head->type[i] == 0){
            printf("%f, ",inst->Attrib[i].f);
        }
        else{
            printf("%d, ",inst->Attrib[i].d);
        }
    }
    printf("\n");
}

void printWeights(int k){
    int i, j;
    for(i = 0; i < k; i++){
        for(j = 0; j < nAttrib; j++){
            printf("%f, ", clusters[i].representative.weight[j]);
        }
        printf("\n");
    }
}

void printAllInstances(){
    int i;
    for(i = 0; i < nInst; i++)
        printInstance(&training[i]);
}

void readFile(char *nameFile){
	FILE *file;
        int i, j, count = 0;
        char c, old = ',', aux[BUFFER];
        char *str = calloc(20, sizeof(char));
	file = fopen(nameFile, "r");
        if(file == NULL){
            printf("Error while trying to open the file");
        }
        else{
            fgets(aux, BUFFER, file);
            nInst = atoi(aux);
            
            fgets(aux, BUFFER, file);
            nAttrib = atoi(aux);
            
            fgets(aux, BUFFER, file);
            nClass = atoi(aux);
                                   
            head = calloc(1, sizeof(Header));
            head->size = nAttrib + nClass;
            head->type = calloc(head->size, sizeof(int));
            head->nFeatures = nAttrib;
            head->nClasses = nClass;

            training = calloc(nInst,(sizeof(Instance)));
            imageX = calloc(nInst,(sizeof(Instance)));
            oldImage = calloc(nInst,(sizeof(Instance)));
            for(i = 0; i < nInst; i++){
                training[i].Attrib = calloc(head->size, sizeof(Value));
                imageX[i].Attrib = calloc(head->size, sizeof(Value));
                oldImage[i].Attrib = calloc(head->size, sizeof(Value));
            }
            
            for(i = 0; i < head->size; i++){
                while((c = fgetc(file)) != '\n' && c != ',') str[count++] = c;
                str[count] = '\0';
                head->type[i] = atoi(str);                
                str[0] = '\0';
                count = 0;                
            }
            
            for(i = 0; i < nInst; i++){                
                training[i].list = calloc(1, sizeof(AssignList));
                training[i].list->flag = 0;                
                training[i].nAttrib = nAttrib;
                training[i].index = i;
                imageX[i].nAttrib = nAttrib;
                imageX[i].class = -1;
                imageX[i].index = -1;
                imageX[i].weight = calloc(nAttrib, sizeof(float));
                imageX[i].auxWeight = calloc(nAttrib, sizeof(float));
                oldImage[i].nAttrib = nAttrib;
                oldImage[i].class = -1;
                oldImage[i].index = -1;
                oldImage[i].weight = calloc(nAttrib, sizeof(float));
                oldImage[i].auxWeight = calloc(nAttrib, sizeof(float));
                for(j = 0; j < head->size; j++){
                    while((c = fgetc(file)) != EOF && c != ',' && c != '\n') str[count++] = c;
                    str[count] = '\0';
                    if(head->type[j] == 0){
                        if(strcmp(str,"?\0") == 0)
                            training[i].Attrib[j].d = MISSING;
                        else{
                            training[i].Attrib[j].f = atof(str);
                            training[i].Attrib[j].d = -3;
                        }
                    }
                    else{
                        if(strcmp(str,"?\0") == 0)
                            training[i].Attrib[j].d = MISSING;
                        else
                            training[i].Attrib[j].d = atoi(str);                                    
                    }
                    if(nClass == 1)
                        if(j == nAttrib)
                            training[i].class = atoi(str);

                    str[0] = '\0';
                    count = 0;
                }
            }                  
        }
	fclose(file);  
}

void negativeArray(int* array, int size){
    int i;
    for(i = 0; i < size; i++)
        array[i] = -1;
}

void cleanArray(int* array, int size){
    int i;
    for(i = 0; i < size; i++)
        array[i] = 0;
}

void copyArray(int *A, int *B, int size){
    int i;
    for(i = 0; i < size; i++)
        B[i] = A[i];
}

void copyFloatArray(float* a, float* b, int l){
    int i, j;
    for(i = 0; i < l; i++)
        b[i] = a[i];
}

float absFloat(float x){
    if(x<0)
        return (-1)*x;
    return x;
}

int isInto(int dat, int *array, int size){
    int i;
    for(i = 0; i < size; i++){
        if(array[i] == dat)
            return 1;
    }
    return 0;
}

void copy(Instance *a, Instance *b){
    int i;
    for(i = 0; i < nAttrib; i++){
        if(a->Attrib[i].d == MISSING)
            b->Attrib[i].d = MISSING;
        else if(head->type[i] == 0)        
            b->Attrib[i].f = a->Attrib[i].f;
        else
            b->Attrib[i].d = a->Attrib[i].d;
    }
    b->class = a->class;
    b->index = a->index;
    b->nAttrib = a->nAttrib;
}

int equals(Instance *a, Instance *b){
    int i, counter = 0;
    for(i = 0; i < nAttrib; i++){
        if(head->type[i] == 0){
            if(a->Attrib[i].d == MISSING && b->Attrib[i].d == MISSING)
                counter++;
            else if(a->Attrib[i].f == b->Attrib[i].f)
                counter++;
        }
        else
            if(a->Attrib[i].d == b->Attrib[i].d)
                counter++;
    }
    if(counter == nAttrib)
        return 1;
    return 0;
}

int isIntoCluster(int index, Cluster *cluster){
    int i;
    for(i = 0; i < cluster->size; i++){
        if(cluster->instances[i] == index)
            return 1;
    }
    return 0;
}
/*----------------------------------*/

/*All Distance Functions*/

/*Euclidean*/

float EuclideanWeighted(Instance *a, Instance *b, float *w, float beta){
    float aux = 0;
    int i;
    for(i = 0; i < nAttrib; i++){        
        aux += pow(w[i], beta) * pow(a->Attrib[i].f - b->Attrib[i].f, 2);        
    }
    return aux;
}

/*----------------------------------*/

float objetiveCriterion(float beta, int flag){
    int i;
    float res = 0;
    if(flag == 0){
        for(i = 0; i < nInst; i++)
            res += EuclideanWeighted(&training[i], &imageX[i], imageX[i].weight, beta);        
    }
    else{
        for(i = 0; i < nInst; i++)
            res += EuclideanWeighted(&training[i], &imageX[i], imageX[i].auxWeight, beta);        
    }
    return res;
}

void calculateAuxWeightsOfImage(Instance *inst){
    int i, j;
    float res = 0;
    for(i = 0; i < nAttrib; i++){
        for(j = 0; j < inst->list->size; j++){
            res += clusters[inst->list->A[j]].representative.auxWeight[i];
        }
        imageX[inst->index].auxWeight[i] = (float)(res/(float)(inst->list->size));
        res = 0;
    }        
}

void weighting(int k, float beta){
    int i, j, l, m, ind;
    float aux1 = 0, aux2 = 0;    
    for(i = 0; i < k; i++){
        if(clusters[i].size > 0){
            for(m = 0; m < nAttrib; m++){
                for(l = 0; l < clusters[i].size; l++){
                    ind = clusters[i].instances[l];
                    if(training[ind].list->size == 1)
                        aux1 += pow(training[ind].Attrib[m].f - clusters[i].representative.Attrib[m].f, 2);  
                }
                aux1 = pow(aux1, (float) 1 / ((float) 1 - beta));
                //aux1 = pow(aux1, (float) 1 / (beta - (float) 1));
                clusters[i].representative.auxWeight[m] = aux1;
                aux2 += aux1;                
                aux1 = 0;
            }
            for(j = 0; j < nAttrib; j++){                
                clusters[i].representative.auxWeight[j] = (clusters[i].representative.auxWeight[j] / aux2);
            }
            aux2 = 0;
        }
        
    }
    for(i = 0; i < nInst; i++)
        calculateAuxWeightsOfImage(&training[i]);
    aux1 = objetiveCriterion(beta, 0);
    aux2 = objetiveCriterion(beta, 1);    
    if(aux2 < aux1){
        for(i = 0; i < k; i++)
            copyFloatArray(clusters[i].representative.auxWeight, clusters[i].representative.weight, nAttrib);        
    }    
}

float calculateXic(int instanceIndex, int featureIndex, int exclude, float beta){
    int i, aux;
    float avgC = 0, avgL = 0, res;
    for(i = 0; i < training[instanceIndex].list->size; i++){
        aux = training[instanceIndex].list->A[i];
        if(aux != exclude){
            avgC += clusters[aux].representative.Attrib[featureIndex].f * pow(clusters[aux].representative.weight[featureIndex], beta);            
        }
        avgL += pow(clusters[aux].representative.weight[featureIndex], beta);
    }
    res = (training[instanceIndex].Attrib[featureIndex].f - (avgC / avgL)) * (avgL / pow(clusters[exclude].representative.weight[featureIndex], beta));    
    return res;
}

void calculateCentroids(int k, float beta){
    int i, j, l, m, ind;
    float val1 = 0, val2 = 0, val3 = 0, aux = 0, resU = 0, resD = 0, xi;
        
    for(i = 0; i < k; i++){        
        if(clusters[i].size != 0){
            for(j = 0; j < nAttrib; j++){
                for(l = 0; l < clusters[i].size; l++){
                    ind = clusters[i].instances[l];                    
                    xi = calculateXic(ind, j, i, beta);
                    val1 = xi * pow(imageX[ind].weight[j], beta);
                    for(m = 0; m < training[ind].list->size; m++)
                        aux += pow(clusters[training[ind].list->A[m]].representative.weight[j], beta);
                    val3 = pow(aux, 2);
                    val2 = pow(imageX[ind].weight[j], beta);
                    resU += val1 / val3;
                    resD += val2 / val3;
                    aux = 0;
                    val1 = 0;
                    val2 = 0;
                    val3 = 0;
                }                                
                clusters[i].representative.Attrib[j].f = resU / resD;;
                resU = 0;
                resD = 0;                
            }            
        }        
        clusters[i].representative.class = i;
    }        
}

void image(int *A, int size, Instance *p, float beta){
    int i, j;
    float aux1 = 0, aux2 = 0, aux3 = 0, auxW = 0;    
    for(i = 0; i < nAttrib; i++){        
        for(j = 0; j < size; j++){
            aux3 = pow(clusters[A[j]].representative.weight[i], beta);
            aux1 += aux3 * clusters[A[j]].representative.Attrib[i].f;
            aux2 += aux3;
            auxW += clusters[A[j]].representative.weight[i];
        }
        p->Attrib[i].f = aux1 / aux2;
        p->weight[i] = auxW / size;
        aux1 = 0;
        aux2 = 0;
        auxW = 0;
    }
}

int assign(int index, int k, float beta){
    int i, z = 0, count = 0, oldCount, res = 0, resAux = 0;       
    Instance *tImage, *oldTImage;    
    float dis1, dis2, disG, disOldG;
    int *A, *R, *W;
    A = training[index].oldList->A;
    W = training[index].oldList->W;
    R = training[index].R;
    tImage = &imageX[index];
    oldTImage = &oldImage[index];
    
    negativeArray(A, k);
    cleanArray(R, k);
    cleanArray(W, k);
    
    dis1 = EuclideanWeighted(&training[index], &clusters[z].representative, clusters[z].representative.weight, beta);
    for(i = 1; i < k; i++){
        dis2 = EuclideanWeighted(&training[index], &clusters[i].representative, clusters[i].representative.weight, beta);
        if(dis2 < dis1){
            dis1 = dis2;
            z = i;
        }            
    }    
    A[count++] = z;
    R[z] = -1;
    image(A, count, tImage, beta);    
    disG = EuclideanWeighted(&training[index], tImage, tImage->weight, beta);    
    do{        
        dis1 = 3.40e38;
        for(i = 0; i < k; i++){
            if(R[i] != -1){
                dis2 = EuclideanWeighted(&training[index], &clusters[i].representative, clusters[i].representative.weight, beta);
                if(dis2 < dis1){
                    dis1 = dis2;
                    z = i;
                }
            }
        }
        
        oldCount = count;        
        A[count] = z;
        image(A, count + 1, tImage, beta);
        dis2 = EuclideanWeighted(&training[index], tImage, tImage->weight, beta);
        if(dis2 < disG){ 
            R[z] = -1;
            count++;
            disG = dis2;
        }        
    }while(count != oldCount && count < k);  
    //negativeArray(W, k);
    for(i = 0; i < k; i++)
        if(R[i] == -1)
            W[i] = 1;    
    
    
    if(training[index].list->flag == 1){
        image(training[index].list->A, training[index].list->size, oldTImage, beta);
        disOldG = EuclideanWeighted(&training[index], oldTImage, oldTImage->weight, beta);
    }
    else{
        disOldG = 3.40e38;
    }
    
    if(training[index].list->flag == 1){
        for(i = 0; i < k; i++)
            if(W[i] == 1)
                res ++;
        
        for(i = 0; i < k; i++)
            if(training[index].list->W[i] == 1)
                resAux ++;        
    }
    else{                
        res = 0;
        resAux = 1;
    }
    
    if(disG < disOldG){
        A = training[index].list->A;
        training[index].list->A = training[index].oldList->A;
        training[index].oldList->A = A;
        
        W = training[index].list->W;
        training[index].list->W = training[index].oldList->W;
        training[index].oldList->W = W;
        
        training[index].list->flag = 1;
        training[index].list->size = count;        
    }
    image(training[index].list->A, training[index].list->size, tImage, beta);
            
    if(res == resAux)
        return 1;
    return 0;
}

int assignAll(int k, float beta){
    int i, j, res = 0;
    int *aux;
    
    aux = calloc(k, sizeof(int));
        
    for(i = 0; i < nInst; i++){
        res += assign(i, k, beta);
        for(j = 0; j < k; j++){
            if(training[i].list->W[j] == 1)
                clusters[j].instances[aux[j]++] = i;
        }
    }

    for(i = 0; i < k; i++){
        clusters[i].size = aux[i];        
    }
        
    free(aux);
    return res;
}

void init(int k){
    srand (time(NULL));
    int i, j, *indexes, random;
    float cons;
    indexes = calloc(k, sizeof(int));
    clusters = calloc(k, sizeof(Cluster));
    for(i = 0; i < k; i++){
        clusters[i].representative.Attrib = calloc(nAttrib, sizeof(Value));
        clusters[i].instances = calloc(nInst, sizeof(int));
    }
    
    for(i = 0; i < k; i++)
        indexes[i] = -1;
    
    for(i = 0; i < k; i++){
        random = rand() % nInst;
        while(isInto(random, indexes, k) == 1)
            random = rand() % nInst;
        indexes[i] = random;
    }
    
    
    for(i = 0; i < k; i++)
        copy(&training[indexes[i]], &clusters[i].representative);    
    
    cons = (float)((float) 1 / (float)(nAttrib));
    for(i = 0; i < k; i++){
        clusters[i].representative.weight = calloc(nAttrib, sizeof(float));
        clusters[i].representative.auxWeight = calloc(nAttrib, sizeof(float));
        for(j = 0; j < nAttrib; j++){
            clusters[i].representative.weight[j] = cons;
            clusters[i].representative.auxWeight[j] = cons;
        }
    }
    
    for(i = 0; i < nInst; i++){
        training[i].oldList = calloc(1, sizeof(AssignList));
        training[i].oldList->A = calloc(k, sizeof(int));
        training[i].oldList->W = calloc(k, sizeof(int));
        training[i].list->A = calloc(k, sizeof(int));
        training[i].list->W = calloc(k, sizeof(int));
        training[i].R = calloc(k, sizeof(int));
    }
}

void initNotRandom(int k, int seed){
    srand (seed);
    int i, j, *indexes, random;
    float cons;
    indexes = calloc(k, sizeof(int));
    clusters = calloc(k, sizeof(Cluster));
    for(i = 0; i < k; i++){
        clusters[i].representative.Attrib = calloc(nAttrib, sizeof(Value));
        clusters[i].instances = calloc(nInst, sizeof(int));
    }
    
    for(i = 0; i < k; i++)
        indexes[i] = -1;
    
    for(i = 0; i < k; i++){
        random = rand() % nInst;
        while(isInto(random, indexes, k) == 1)
            random = rand() % nInst;
        indexes[i] = random;
    }
    
    
    for(i = 0; i < k; i++)
        copy(&training[indexes[i]], &clusters[i].representative);    
    
    cons = (float)((float) 1 / (float)(nAttrib));
    for(i = 0; i < k; i++){
        clusters[i].representative.weight = calloc(nAttrib, sizeof(float));
        clusters[i].representative.auxWeight = calloc(nAttrib, sizeof(float));
        for(j = 0; j < nAttrib; j++){
            clusters[i].representative.weight[j] = cons;
            clusters[i].representative.auxWeight[j] = cons;
        }
    }
    
    for(i = 0; i < nInst; i++){
        training[i].oldList = calloc(1, sizeof(AssignList));
        training[i].oldList->A = calloc(k, sizeof(int));
        training[i].oldList->W = calloc(k, sizeof(int));
        training[i].list->A = calloc(k, sizeof(int));
        training[i].list->W = calloc(k, sizeof(int));
        training[i].R = calloc(k, sizeof(int));
    }
}

void WOKM(int k, float beta){
    int band = 0, count = 0, aux = -1;
        
    start = time(NULL);
    while(band == 0){
        count = assignAll(k, beta);
        if(count == aux)
            band = 1;
        else{
            calculateCentroids(k, beta);
            weighting(k, beta);
        }
        aux = count;
		iterations++;
    }
    end = time(NULL);
}
/*----------------------------------*/

/*argv[0] = Program Name
 *argv[1] = Training File
 *argv[2] = k clusters
 *argv[3] = beta(must be > 1) 
 *argv[4] = seed (optional)
 */

int main(int argc, char** argv) {
    int k, i, j, *index;
    float beta;
    FILE *fileOut;
    readFile(argv[1]);
    k = atoi(argv[2]);
    beta = atof(argv[3]);
    
    if(argc > 4){
        initNotRandom(k, atoi(argv[3]));
    }
    else
        init(k);
    
    
    
    WOKM(k, beta);
    
    fileOut = fopen("WOKM-Output.txt","w");
    for(i = 0; i < k; i++){
        for(j = 0; j < clusters[i].size - 1; j++){
            fprintf(fileOut, "%d", clusters[i].instances[j]);
            fputc(',', fileOut);
        }
        fprintf(fileOut, "%d\n", clusters[i].instances[j]);
    }
    fprintf(fileOut, "-2\n");
    for(i = 0; i < k; i++){
        for(j = 0; j < clusters[i].representative.nAttrib; j++){
            fprintf(fileOut, "%f,",clusters[i].representative.Attrib[j].f);
        }
        fprintf(fileOut, "%d\n", clusters[i].representative.class);
    }
    fprintf(fileOut, "%f\n", difftime(end, start));
    fprintf(fileOut, "%d\n", iterations);
        
    fclose(fileOut);
    
    
    
    return (EXIT_SUCCESS);
}
