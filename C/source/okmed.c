/* 
    OKMED Implementation

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
Instance *training, *image, *oldImage;
Cluster *clusters;
int nAttrib = 0, nInst = 0, nClass = 0, iterations = 0;
float **dis;
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
            image = calloc(nInst,(sizeof(Instance)));
            oldImage = calloc(nInst,(sizeof(Instance)));
            for(i = 0; i < nInst; i++){
                training[i].Attrib = calloc(head->size, sizeof(Value));
                image[i].Attrib = calloc(head->size, sizeof(Value));
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
                image[i].nAttrib = nAttrib;
                image[i].class = -1;
                image[i].index = -1;
                oldImage[i].nAttrib = nAttrib;
                oldImage[i].class = -1;
                oldImage[i].index = -1;
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
                        if(j == nAttrib-1)
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
    int i, x, y;
    if(a->index == b->index)
        return 0;
    if(a->index > b->index){
        x = a->index - 1;
        y = b->index;
    }
    else{
        x = b->index - 1;
        y = a->index;
    }
    if(dis[x][y] == 0){
        for(i = 0; i < nAttrib; i++){        
            aux += pow(a->Attrib[i].f - b->Attrib[i].f, 2);        
        }
        dis[x][y] = aux;
    }
    return dis[x][y];
}

/*----------------------------------*/


/*Overlap K-MEDOIDS Clustering Algorithm*/
float inertia(int clusterIndex){
    int i;
    float res = 0;
    if(clusters[clusterIndex].size > 0)
        for(i = 0; i < clusters[clusterIndex].size; i++){
            res += Euclidean(&training[clusters[clusterIndex].instances[i]], &image[clusters[clusterIndex].instances[i]]);
        }
    return res;
}

void imageMedoid(int *A, int size, Instance *p){
    int i, j, index = -1;
    float distance = 0, aux = 3.40e38;
    if(size > 1){
        for(i = 0; i < nInst; i++){
            for(j = 0; j < size; j++)
                distance += Euclidean(&clusters[A[j]].representative, &training[i]);
            
            if(distance < aux){
                aux = distance;
                index = i;
            }
            distance = 0;
        }
        copy(&training[index], p);
    }
    else{
        copy(&clusters[A[0]].representative, p);
    }
    
}

int Medoid(int indexCluster, int k){
    int i, j, l, aux;
    Instance temp;
    temp.Attrib = calloc(nAttrib, sizeof(Value));
    float inertia2 = 0;
    float inertia1 = inertia(indexCluster);
    aux = clusters[indexCluster].representative.index;     
    for(i = 0; i < k; i++){
        for(j = 0; j < clusters[indexCluster].size; j++){
            if(training[clusters[indexCluster].instances[j]].list->size == (i + 1)){
                copy(&training[clusters[indexCluster].instances[j]], &clusters[indexCluster].representative);                
                for(l = 0; l < clusters[indexCluster].size; l++){
                    imageMedoid(training[clusters[indexCluster].instances[l]].list->A, training[clusters[indexCluster].instances[l]].list->size, &temp);
                    inertia2 += Euclidean(&temp, &training[clusters[indexCluster].instances[l]]);
                }
                if(inertia2 < inertia1){
                    return clusters[indexCluster].instances[j];
                }
                else{
                    inertia2 = 0;
                }
            }
        }
    }
    copy(&training[aux], &clusters[indexCluster].representative);
    return -1;
}

void searchMedoids(int k){
    int i, aux = 0;
    for(i = 0; i < k; i++){        
        aux = Medoid(i, k);
    }
}


int assign(int index, int k){
    int i, z = 0, count = 0, oldCount, res = 0, resAux = 0;       
    Instance *imageX, *oldImageX;    
    float dis1, dis2, disG, disOldG;
    int *A, *R, *W;    
    A = training[index].oldList->A;
    W = training[index].oldList->W;
    R = training[index].R;
    imageX = &image[index];
    oldImageX = &oldImage[index];
    
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
    imageMedoid(A, count, imageX);    
    disG = Euclidean(&training[index], imageX);    
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
        imageMedoid(A, count + 1, imageX);
        dis2 = Euclidean(&training[index], imageX);
        if(dis2 < disG){ 
            R[z] = -1;
            count++;
            disG = dis2;
        }        
    }while(count != oldCount && count < k);  
    for(i = 0; i < k; i++)
        if(R[i] == -1)
            W[i] = 1;    
    
    if(training[index].list->flag == 1){
        imageMedoid(training[index].list->A, training[index].list->size, oldImageX);
        disOldG = Euclidean(&training[index], oldImageX);
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
    imageMedoid(training[index].list->A, training[index].list->size, imageX);
    
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
    
    dis = calloc(nInst - 1, sizeof(float*));
    for(i = 0; i < nInst - 1; i++){
        dis[i] = calloc(i + 1, sizeof(float));
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
    
    dis = calloc(nInst - 1, sizeof(float*));
    for(i = 0; i < nInst - 1; i++){
        dis[i] = calloc(i + 1, sizeof(float));
    }
    
}

void OKMED(int k){
    int band = 0, count = 0, aux = -1;
        
    //init(k);
    start = time(NULL);
    while(band == 0){        
        count = assignAll(k);        
        if(count == aux)
            band = 1;
        else
            searchMedoids(k);
        aux = count;    
		iterations++;
    }
    end = time(NULL);
}

/*----------------------------------*/

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
    
    
    OKMED(k);
    
    fileOut = fopen("OKMED-Output.txt","w");
    for(i = 0; i < k; i++){
        for(j = 0; j < clusters[i].size - 1; j++){
            fprintf(fileOut, "%d", clusters[i].instances[j]);
            fputc(',', fileOut);
        }
        fprintf(fileOut, "%d\n", clusters[i].instances[j]);
    }
    fprintf(fileOut, "-3\n");
    for(i = 0; i < k; i++){        
        fprintf(fileOut, "%d\n", clusters[i].representative.index);
    }
    fprintf(fileOut, "%f\n", difftime(end, start));
    fprintf(fileOut, "%d\n", iterations);
        
    fclose(fileOut);
    
    
    
    return (EXIT_SUCCESS);
}
