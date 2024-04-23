#include "aco.h"
#include <omp.h>

int rand_int(int min, int max) {
    assert(min <= max);
    return rand() % (max - min + 1) + min;
}

bool contains(size_t* arr, size_t size, size_t val) {
    for (size_t i = 0; i < size; i++) {
        if (arr[i] == val) {
            return 1;
        }
    }
    return 0;
}

size_t* setdiff1d(size_t* a, size_t* b, size_t size_a, size_t size_b, size_t* size_r) {
    size_t count = 0;
    for (size_t i = 0; i < size_a; i++) {
        if (!contains(b, size_b, a[i])) {
            count++;
        }
    }
    size_t* result = malloc((size_t)(sizeof(size_t) * count));
    size_t index = 0;
    for (size_t i = 0; i < size_a; i++) {
        if (!contains(b, size_b, a[i])) {
            result[index++] = a[i];
        }
    }
    *size_r = &count;
    free(a);
    return result;
}

size_t* nonzero_index(double* array, size_t size, size_t* size_r) {
    size_t count = 0;
    for (size_t i = 0; i < size; i++) {
        if (array[i] != 0) {
            count++;
        }
    }
    size_t* result = malloc((size_t)(sizeof(size_t) * count));
    size_t index = 0;
    for (size_t i = 0; i < size; i++) {
        if (array[i] != 0) {
            result[index++] = i;
        }
    }
    *size_r = &count;
    return result;
}

double sum(double* arr, size_t size) {
    double sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

void power(double* arr, size_t size, double exponent) {
    for (size_t i = 0; i < size; i++) {
        arr[i] = pow(arr[i], exponent);
    }
}

inline void multiply_elements(double* a, double* b, size_t size) {
    assert(a != NULL && b != NULL);
    for (size_t i = 0; i < size; i++) {
        a[i] *= b[i];
    }
}

size_t random_choice(double* probabilities, size_t size) {
    double random_value = (double)rand() / RAND_MAX * sum(probabilities, size);
    double cumulative_probability = 0;

    for (size_t i = 0; i < size; i++) {
        cumulative_probability += probabilities[i];
        if (random_value <= cumulative_probability) {
            return i;
        }
    }
    return size - 1;
}

void init_rand(time_t t) {
    srand(t);
}


size_t* step(double** closeness_matrix, double** pheromone_matrix, const size_t node_count, const size_t ant_count,
    const double A, const double B, const double Q, const double evap, double* bpl) {

    size_t* better_path = malloc((size_t)(sizeof(size_t) * node_count));
    double better_path_len = LONG_MAX;

    // simulate ant colony
    /*size_t** paths = (size_t**)malloc(ant_count * sizeof(size_t*));
    for (size_t i = 0; i < ant_count; i++)
        paths[i] = (size_t*)malloc(node_count * sizeof(size_t));
    double* lens = malloc((double)(sizeof(double) * ant_count));*/

    // число потоков 
    //omp_set_num_threads(g_nNumberOfThreads);
    int ant;
    #pragma omp parallel for private(ant) shared(better_path) shared(better_path_len)
    for (ant = 0; ant < ant_count; ant++) {
        size_t* path = malloc((size_t)(sizeof(size_t) * node_count));
        path[0] = rand_int(0, node_count - 1);
        size_t p = 0;

        for (size_t i = 1; i < node_count; i++) {
            size_t* size_enable;
            size_t* enable = nonzero_index(closeness_matrix[path[i - 1]], node_count, &size_enable);

            enable = setdiff1d(enable, path, *size_enable, i, &size_enable);

            if (*size_enable == 0) break;

            size_t size = *size_enable;
            double* n = (double*)malloc((size) * sizeof(double)); // близость
            double* t = (double*)malloc((size) * sizeof(double)); // феромоны

            for (size_t j = 0; j < size; j++) {
                n[j] = closeness_matrix[path[i - 1]][enable[j]];
                t[j] = pheromone_matrix[path[i - 1]][enable[j]];
            }
            power(n, size, B);
            power(t, size, A);

            multiply_elements(n, t, size); // результат в n
            free(t);

            size_t index = random_choice(n, size);
            free(n);
            path[i] = enable[index];
            p++;
            free(enable);
        }

        // validator
        if ((p < node_count - 1) || (closeness_matrix[path[node_count - 1]][path[0]] == 0)) {
            continue;
        }

        // calculate path len
        double path_len = 0;
        for (size_t i = 0; i < node_count - 1; i++) {
            path_len += 200 / closeness_matrix[path[i]][path[i + 1]];
        }   path_len += 200 / closeness_matrix[path[0]][path[node_count - 1]];

        if (better_path_len > path_len) {
            for (size_t r = 0; r < node_count; r++) {
                better_path[r] = path[r];
            }
            better_path_len = path_len;
        } 
        /*for (size_t r = 0; r < node_count; r++) {
            paths[ant][r] = path[r];
        }
        lens[ant] = path_len;*/

        free(path);
    }
    

    /*size_t index_min = 0;
    for (size_t i = 0; i < ant_count; i++) {
        if (better_path_len > lens[i]) {
            better_path_len = lens[i]; 
            index_min = i;
        }
    }
    for (size_t i = 0; i < node_count; i++)
        better_path[i] = paths[index_min][i];*/


    // evaporation
    double e = 1 - evap;
    for (size_t i = 0; i < node_count; i++)
        for (size_t j = 0; j < node_count; j++)
            pheromone_matrix[i][j] *= e;

    // add ph
    double ph = Q / better_path_len;
    for (size_t i = 0; i < node_count - 1; i++) {
        pheromone_matrix[better_path[i]][better_path[i + 1]] += ph;
        pheromone_matrix[better_path[i + 1]][better_path[i]] += ph;
    }
    pheromone_matrix[better_path[0]][better_path[node_count - 1]] += ph;
    pheromone_matrix[better_path[node_count - 1]][better_path[0]] += ph;

    // max-min realization 
    for (size_t i = 0; i < node_count; i++)
        for (size_t j = 0; j < node_count; j++)
            if (pheromone_matrix[i][j] > 1) 
                pheromone_matrix[i][j] = 1;

    *bpl = better_path_len;
    return better_path;
}

//size_t* run(const size_t k, double** closeness_matrix, double** pheromone_matrix, const size_t node_count, const size_t ant_count,
//           const double A, const double B, const double Q, const double evap, double* bpl) {
//    size_t* best_path = malloc((size_t)(sizeof(size_t) * node_count));
//    double best_path_len = LONG_MAX;
//
//    double path_len;
//    for (size_t i = 0; i < k; i++) {
//        size_t* temp = step(closeness_matrix, pheromone_matrix, node_count, ant_count, A, B, Q, evap, &path_len);
//        if (path_len < best_path_len) {
//            best_path_len = path_len;
//            for (size_t r = 0; r < node_count; r++) {
//                best_path[r] = temp[r];
//            }
//        }
//    }
//    *bpl = best_path_len;
//    for (size_t r = 0; r < node_count; r++) {
//        printf("%d ", best_path[r]);
//    }
//    return best_path;
//}
