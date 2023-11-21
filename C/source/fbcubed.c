/* 
 * File:   FBCubed.c
 * Author: Aldair
 *
 * Created on 1 de mayo de 2013, 10:51 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define BUFFER 2048000

typedef struct {
    int overlapedClass;
    int overlapedCluster;
	int index;
    int *classes;
    int *clusters;
} Data;

typedef struct {
    int *index;
    int size;
} Array;


Data *instances;
Array *clusters, *classes;
float FBcubed, BCRecall, BCPrecision;
int nAttrib = 0, nInst = 0, nClass = 0, nCluster = 0, totalSize;

void readLabeledFile(char *nameFile){
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
        
        fgets(aux, BUFFER, file);

        totalSize = nClass + nAttrib;
        instances = calloc(nInst, sizeof(Data));
        for(i = 0; i < nInst; i++){
			instances[i].index = i;
            instances[i].overlapedClass = 0;
            instances[i].overlapedCluster = 0;
            instances[i].classes = calloc(nClass, sizeof(int));
        }

        for(i = 0; i < nInst; i++){
            for(j = 0; j < totalSize; j++){
                while((c = fgetc(file)) != EOF && c != ',' && c != '\n') str[count++] = c;
                str[count] = '\0';
                if(j >= nAttrib)
                    instances[i].classes[j - nAttrib] = atoi(str);

                str[0] = '\0';
                count = 0;
            }
        }            
    }
    fclose(file);
}

void readClusterFile(char *nameFile){
    FILE *file;
    int i, j, count = 0, flag = 1, tmp, *size;
    char c, old = ',', aux[BUFFER];
    char *str = calloc(20, sizeof(char));
    file = fopen(nameFile, "r");
    if(file == NULL){
        printf("Error while trying to open the file");
    }
    else{
        while(flag){
            fgets(aux, BUFFER, file);
            if(aux[0] != '-')
                count++;
            else
                flag = 0;
        }
        
        nCluster = count;
        for(i = 0; i < nInst; i++)
            instances[i].clusters = calloc(nCluster, sizeof(int));
        
        rewind(file);
        count = 0;
        size = calloc(nCluster, sizeof(int));
        
        for(i = 0; i < nCluster; i++){
            while((c = fgetc(file)) != '\n' && c != EOF){
                if(old == ',')
                    count++;
                old = c;
            }
            size[i] = count;
            count = 0;
            old = ',';
        }
        
        rewind(file);
        
        for(i = 0; i < nCluster; i++){
            for(j = 0; j < size[i]; j++){            
                while((c = fgetc(file)) != '\n' && c != ',' && c != EOF) str[count++] = c;
                str[count] = '\0';
                tmp = atoi(str);
                instances[tmp].clusters[i] = 1;
                str[0] = '\0';
                count = 0;
            }
        }        
    }
    fclose(file);
}

int min(int a, int b){
    if(a < b)
        return a;
    return b;
}

int isInto(int dat, int *array, int size){
    int i;
    for(i = 0; i < size; i++){
        if(array[i] == dat)
            return 1;
    }
    return 0;
}

int isIntoArray(int dat, Array *a){
    return isInto(dat, a->index, a->size);
}

int sharedClusters(Data *a, Data *b){
    int i, count = 0;
    
    for(i = 0; i < nCluster; i++){
        if(a->clusters[i] == 1 && b->clusters[i] == 1)
            count++;
    }
    
    return count;
}

int sharedClasses(Data *a, Data *b){
    int i, count = 0;
    
    for(i = 0; i < nClass; i++)
        if(a->classes[i] == 1 && b->classes[i] == 1)
            count++;
    
    return count;
}

float multPre(Data *a, Data *b){
    int c1, c2;
    
    c2 = sharedClusters(a, b);
    c1 = min(c2, sharedClasses(a, b));
    
    return (float)((float)c1/(float)c2);
}

float multRec(Data *a, Data *b){
    int c1, c2;
    
    c2 = sharedClasses(a, b);
    c1 = min(sharedClusters(a, b), c2);
    	
    return (float)((float)c1/(float)c2);
}

float BcubedPre(Data *o){
    int i, count = 0;
    float res = 0;
    
    for(i = 0; i < nInst; i++){
        if(sharedClusters(o, &instances[i]) > 0){
            res += multPre(o, &instances[i]);
            count++;
        }
    }
    
    return res/(float)count;
}

float BcubedRec(Data *o){
    int i, count = 0;
    float res = 0;
    
    for(i = 0; i < nInst; i++){
        if(sharedClasses(o, &instances[i]) > 0){
            res += multRec(o, &instances[i]);
            count++;
        }
    }
	//if(o->index == 1382)
		//printf("BRecNum: %f ---- BRecDen: %d\n", res, count);
    if(count == 0)
        printf("%d\n", o->index);
    
    return res/(float)count;
}

void FBC(){
    int i;
    float BcubPre = 0, BcubRec = 0;
    
    for(i = 0; i < nInst; i++){
        BcubPre += BcubedPre(&instances[i]);
        BcubRec += BcubedRec(&instances[i]);
        //printf("Index: %d ---- Pre: %f ----- Rec: %f\n", i, BcubPre, BcubRec);
    }
    printf("clu: %f, cla: %f, nI: %d\n", BcubPre, BcubRec, nInst);
    BcubPre = BcubPre / (float)nInst;
    BcubRec = BcubRec / (float)nInst;
    printf("%f--%f->%f\n", BcubPre, BcubRec, ((float)2 * BcubPre * BcubRec) / (BcubPre + BcubRec));
    BCRecall = BcubRec;
    BCPrecision = BcubPre;
    FBcubed = ((float)2 * BcubPre * BcubRec) / (BcubPre + BcubRec);
}


/*
 * 
 */
int main(int argc, char** argv) {
    FILE *fileOut;
    
    readLabeledFile(argv[1]);
    readClusterFile(argv[2]);
    
    FBC();
    fileOut = fopen("FBcubed-Output.txt","w");    
    fprintf(fileOut, "FBcubed Precision: %f\n", BCPrecision);
    fprintf(fileOut, "FBcubed Recall: %f\n", BCRecall);
    fprintf(fileOut, "FBcubed: %f\n", FBcubed);
    
    fclose(fileOut);

    return (EXIT_SUCCESS);
}

