/* 
 * File:   FMeasure.c
 * Author: Argenis Aldair Aroche Villarruel
 *
 * Created on 31 de mayo de 2013, 12:26 AM
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
    int *classes;
    int *clusters;
} Data;

typedef struct {
    int *index;
    int size;
} Array;


Data *instances;
Array *clusters, *classes;
float Precision, Recall, FMeasure, CardinalityR, CardinalityO;
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

void Overlap(){
    int i, j, count = 0;
    
    for(i = 0; i < nInst; i++)
        for(j = 0; j < nCluster; j++){
            if(instances[i].clusters[j] == 1)
                count++;
        }
    
    CardinalityR = (float)count / (float)nInst;
    count = 0;
            
    for(i = 0; i < nInst; i++)
        for(j = 0; j < nClass; j++){
            if(instances[i].classes[j] == 1)
                count++;
        }
    CardinalityO = (float)count / (float)nInst;
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

int NoNr(){
    int i, j, count = 0;
    
    for(i = 0; i < nInst - 1; i++)
        for(j = i + 1; j < nInst; j++)
            if(sharedClusters(&instances[i], &instances[j]) > 0 && sharedClasses(&instances[i], &instances[j]) > 0)
                count++;        
        
    return count;
}

int No(){
    int i, j, count = 0, shared;
    
    for(i = 0; i < nInst - 1; i++)
        for(j = i + 1; j < nInst; j++)
            if(sharedClusters(&instances[i], &instances[j]) > 0)
                count++;
    
    return count;
}

int Nr(){
    int i, j, count = 0;
    
    for(i = 0; i < nInst - 1; i++)
        for(j = i + 1; j < nInst; j++)
            if(sharedClasses(&instances[i], &instances[j]) > 0)
                count++;
    
    return count;
}

void FM(){    
    int interCluCla, pairClust, pairClass;
    
    interCluCla = NoNr();
    pairClust = No();
    pairClass = Nr();
    
    Precision = (float)interCluCla / (float)pairClust;
    Recall = (float)interCluCla / (float)pairClass;
    FMeasure = (2 * Precision * Recall) / (Precision + Recall);
}



/*
 * 
 */
int main(int argc, char** argv) {
    FILE *fileOut;
    
    readLabeledFile(argv[1]);
    readClusterFile(argv[2]);
    Overlap();
    FM();
    fileOut = fopen("FMeasure-Output.txt","w");
    
    fprintf(fileOut, "Precision: %f\n", Precision);
    fprintf(fileOut, "Recall: %f\n", Recall);
    fprintf(fileOut, "FMeasure: %f\n", FMeasure);
    fprintf(fileOut, "OverlapR: %f\n", CardinalityR);
    fprintf(fileOut, "OverlapO: %f\n", CardinalityO);
    
    fclose(fileOut);

    return (EXIT_SUCCESS);
}

