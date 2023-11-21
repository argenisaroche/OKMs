/* 
    OKM Implementation

    Author: Argenis Aldair Aroche Villarruel
    Date: February 2013

    This is the implementation used for the comparison in the paper: 
    Study of Overlapping Clustering Algorithms Based on Kmeans through FBcubed Metric
    https://doi.org/10.1007/978-3-319-07491-7_12

    Implementation based on: 
    An extended version of the k-means method for overlapping clustering
    10.1109/ICPR.2008.4761079

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
Instance *training, *gravCenter, *oldGravCenter;
Cluster *clusters;
int nClass = 0, nAttrib = 0, nInst = 0, iterations = 0;
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
            gravCenter = calloc(nInst,(sizeof(Instance)));
            oldGravCenter = calloc(nInst,(sizeof(Instance)));
            for(i = 0; i < nInst; i++){
                training[i].Attrib = calloc(head->size, sizeof(Value));
                gravCenter[i].Attrib = calloc(head->size, sizeof(Value));
                oldGravCenter[i].Attrib = calloc(head->size, sizeof(Value));
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
                gravCenter[i].nAttrib = nAttrib;
                gravCenter[i].class = -1;
                gravCenter[i].index = -1;
                oldGravCenter[i].nAttrib = nAttrib;
                oldGravCenter[i].class = -1;
                oldGravCenter[i].index = -1;
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
                        if(j == head->size - 1)
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

float Euclidean(Instance *a, Instance *b){
    float aux = 0;
    int i;
    for(i = 0; i < nAttrib; i++){        
        aux += pow(a->Attrib[i].f - b->Attrib[i].f, 2);        
    }
    return sqrt(aux);
}

/*----------------------------------*/



float mc(AssignList *list, int zj, int v){
    int i;
    float sum = 0;  
    if(list->size > 1){
        for(i = 0; i < list->size; i++){        
            if(list->A[i] != zj)
                sum += clusters[list->A[i]].representative.Attrib[v].f;
        }
        return sum;
    }    
    return 0;
}

void calculateCentroids(int k){
    int i, j, l, aux = 0, di;
    float val1 = 0, val2 = 0, res = 0, xi;
    for(i = 0; i < k; i++){        
        if(clusters[i].size != 0){
            for(j = 0; j < nAttrib; j++){
                for(l = 0; l < clusters[i].size; l++){
                    di = training[clusters[i].instances[l]].list->size;
                    xi = (di * training[clusters[i].instances[l]].Attrib[j].f) - (mc(training[clusters[i].instances[l]].list, i, j));
                    val1 += (float)1 / (pow(di, 2));
                    val2 += (float)1 / (pow(di, 2)) * xi;
                }
                res = ((float)1 / val1) * val2;
                clusters[i].representative.Attrib[j].f = res;
                aux = 0;
                val1 = 0;
                val2 = 0;
                res = 0;
            }
        }
        clusters[i].representative.class = i;
    }
}

void gravityCenter(int *A, int size, Instance *p){
    int i, j;
    float mean = 0;    
    for(i = 0; i < nAttrib; i++){        
        for(j = 0; j < size; j++)            
            mean += clusters[A[j]].representative.Attrib[i].f;        
        p->Attrib[i].f = mean/((float)size);
        mean = 0;
    }
}

int assign(int index, int k){
    int i, z = 0, count = 0, oldCount, res = 0, resAux = 0;       
    Instance *gCenter, *oldGCenter;    
    float dis1, dis2, disG, disOldG;
    int *A, *R, *W;
    A = training[index].oldList->A;
    W = training[index].oldList->W;
    R = training[index].R;
    gCenter = &gravCenter[index];
    oldGCenter = &oldGravCenter[index];
    
    negativeArray(A, k);
    cleanArray(R, k);
    cleanArray(W, k);
    
    dis1 = Euclidean(&training[index], &clusters[z].representative);
    for(i = 1; i < k; i++){
        dis2 = Euclidean(&training[index], &clusters[i].representative);
        if(dis2 < dis1){
            dis1 = dis2;
            z = i;
        }            
    }    
    A[count++] = z;
    R[z] = -1;
    gravityCenter(A, count, gCenter);    
    disG = Euclidean(&training[index], gCenter);    
    do{        
        dis1 = 3.40e38;
        for(i = 0; i < k; i++){
            if(R[i] != -1){
                dis2 = Euclidean(&training[index], &clusters[i].representative);
                if(dis2 < dis1){
                    dis1 = dis2;
                    z = i;
                }
            }
        }
        
        oldCount = count;        
        A[count] = z;
        gravityCenter(A, count + 1, gCenter);
        dis2 = Euclidean(&training[index], gCenter);
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
        gravityCenter(training[index].list->A, training[index].list->size, oldGCenter);
        disOldG = Euclidean(&training[index], oldGCenter);
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
            
    if(res == resAux)
        return 1;
    return 0;
}

int assignAll(int k){
    int i, j, res = 0;
    int *aux;
    
    aux = calloc(k, sizeof(int));
        
    for(i = 0; i < nInst; i++){
        res += assign(i, k);
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
    int i, *indexes, random;
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
    int i, *indexes, random;
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
    
    for(i = 0; i < nInst; i++){
        training[i].oldList = calloc(1, sizeof(AssignList));
        training[i].oldList->A = calloc(k, sizeof(int));
        training[i].oldList->W = calloc(k, sizeof(int));
        training[i].list->A = calloc(k, sizeof(int));
        training[i].list->W = calloc(k, sizeof(int));
        training[i].R = calloc(k, sizeof(int));
    }
}
/*----------------------------------*/

void OKM(int k){
    int i, band = 0, count = 0;
    Instance *temp;
    temp = calloc(k, sizeof(Instance));
    for(i = 0; i < k; i++)
        temp[i].Attrib = calloc(head->size, sizeof(Value));
    
    start = time(NULL);
    while(band == 0){
        for(i = 0; i < k; i++)
            copy(&clusters[i].representative, &temp[i]);
        
        assignAll(k);
        calculateCentroids(k);
        for(i = 0; i < k; i++){
            if(equals(&clusters[i].representative, &temp[i]) == 1)
                count++;
        }        
        if(count == k)
            band = 1;
        count = 0;
    }
    end = time(NULL);
}

/*argv[0] = Program Name
 *argv[1] = Training File
 *argv[2] = k clusters
 *argv[3] = seed (optional)
 */

int main(int argc, char** argv) {
    int k, i, j;
    FILE *fileOut;
    readFile(argv[1]);
    k = atoi(argv[2]);
    
    if(argc > 3){
        initNotRandom(k, atoi(argv[3]));
    }
    else
        init(k);
    
    
    OKM(k);
    
    fileOut = fopen("OKM-Output.txt","w");
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
