#include "Renderer.h"
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <vector>
#include <time.h>
#define WINDOW_WIDTH 1000
#define WINDOW_HEIGHT 1000





int main()
{
    
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "CUDA/SFML Ray Tracing");
    sf::Texture image;
    image.create(WINDOW_WIDTH, WINDOW_HEIGHT);
    sf::Sprite frame(image);
    sf::Uint8 *frameBuffer = new sf::Uint8[WINDOW_WIDTH * WINDOW_HEIGHT * 4];
    

    int start = clock();
    initRenderer(WINDOW_HEIGHT, WINDOW_HEIGHT);
    while (window.isOpen())
    {   
        window.clear();

        int time = clock()- start;

        
        renderFrame(frameBuffer, (float)time);
       
        
        //std::cout << clock() - time << "\n";

        image.update(frameBuffer);
        frame.setTexture(image);
        window.draw(frame);


        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                delete[] frameBuffer;
                releaseRenderer();
                window.close();
            }
        }
        window.display();
    }

    
    
    return 0;
}

